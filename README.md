# 变量身份嵌入：给 S-Mamba / iTransformer 的变量 token 加上"身份"

iTransformer 和 S-Mamba 把每个变量的整段历史映射成一个 token，再在变量之间做混合（注意力或双向 Mamba）。两个历史窗口相似的变量会得到相似的 token，模型无法区分"它们是谁"。本仓库给每个变量加一个可学习的身份向量：

```
h_i = Linear(x_i) + e_i ,   i = 1..N      （e_i 从 0 开始学习）
```

并提供在 RTX 4090 上做全尺寸实验所需的全部代码：两种骨架的忠实实现、官方的训练与评估协议、身份嵌入的几种形式与消融、"未见变量"泛化实验、结果汇总和可解释性分析。

**为什么做这个**：在前期 CPU 受控实验（[`experiments_cpu/`](experiments_cpu/README.md)）里，我们比较了稀疏变量混合、顺序鲁棒训练、身份嵌入等改进，唯一在每个种子上都稳定有效的是身份嵌入：Electricity 上 S-Mamba 式模型 −3.9%、Weather 上 −5.5%（预测 336 步时 −3.1%），但在只有 7 个变量的 ETTh1 上无效。那些实验用的是缩小的模型（D=64、2 层），本仓库要在官方规模上检验这个结论。

**已有相关工作**：STID（CIKM 2022）在时空预测中引入了空间身份嵌入；EMAformer（AAAI 2026）给 iTransformer 加了辅助嵌入；ss-Mamba（2025）用语义身份嵌入。身份嵌入本身不是新想法；本仓库的重点是在 S-Mamba 上系统验证它，并回答几个尚未回答的问题：用多大的表（满秩还是低秩）、放在哪里、学习率怎么设，以及遇到训练时没见过的变量怎么办。

**结果与论文存放位置**：本仓库只放代码。所有实验结果（`results/*.jsonl`）、分析和论文统一提交到 [`unforgettablerep/newpaper`](https://github.com/unforgettablerep/newpaper)；本仓库的 `.gitignore` 会忽略本地的 `results/` 目录，跑完后把其中的 JSONL 文件放进 newpaper 仓库的 `results/` 目录。

---

## 1. 方法（`idembed/identity.py`、`idembed/models.py`）

| 选项 | 含义 |
|---|---|
| `--identity none` | 原始模型 |
| `--identity add` | 满秩表 E ∈ R^{N×D}，零初始化 |
| `--identity lowrank --id-rank r` | E = U V，U ∈ R^{N×r}，V ∈ R^{r×D}（V 零初始化），参数更少，作为少变量数据上的正则化 |
| `--id-loc input` / `every` | 只在嵌入层之后加一次 / 在每个编码层之前各加一份 |
| `--id-dropout p` | 训练时以概率 p 把某个变量的身份置零，让模型也学会不依赖身份 |
| `--id-lr-mult m` | 身份参数的学习率是主学习率的 m 倍。身份参数从 0 开始，Adam 每步大约只移动一个学习率的幅度，而官方学习率很小（Weather 5e-5、ETTh1 7e-5） |
| `--train-var-ratio ρ` | 未见变量协议：只用随机选出的 ρ 比例的变量训练和验证，测试时预测全部变量，并分别报告已见 / 未见变量的误差；未见变量使用零身份 |
| `--scan-order shuffle` | （仅 S-Mamba）训练时每一层、每一步都按随机顺序扫描 token，使"在扫描序列中的位置"不再携带变量信息；测试默认按原始顺序 |
| `--eval-orders K` | 训练好的模型额外在测试时把每层的 Mamba 混合输出在 K 个固定随机顺序上取平均，结果记为 `mse_orders` / `mae_orders`（不需要重新训练） |

- 身份只加在变量 token 上，日历特征 token 不加。
- 所有形式在初始化时 E = 0，并且不消耗全局随机数，所以同一种子下加身份前后的模型在初始化时逐位相同（见测试）。两者的差异只来自训练。

## 2. 环境（RTX 4090）

```bash
conda create -n idembed python=3.11 -y && conda activate idembed
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 任一 CUDA 12.x 版本的 PyTorch ≥ 2.1
pip install numpy pandas pytest
# Mamba 的 CUDA 内核（S-Mamba 的快速路径）。需要与 PyTorch 匹配的 CUDA 工具链；
# 编译较慢，也可以从 mamba / causal-conv1d 的 GitHub Releases 下载与 torch / CUDA / Python 版本匹配的预编译 wheel
pip install "causal-conv1d>=1.4.0" --no-build-isolation
pip install "mamba-ssm>=2.2" --no-build-isolation
python scripts/check_env.py
```

`check_env.py` 会：打印 GPU 与 CUDA 信息；把官方 `mamba_ssm.Mamba` 的权重载入本仓库的纯 PyTorch 实现 `MambaTorch`（参数名完全一致）并比较输出，以同时验证两条路径；在 Electricity 规模（B=16，321+4 个 token，d=512，3 层）上测一次 S-Mamba 训练步的耗时和显存。

- `--mamba-impl auto`（默认）：有 GPU 且装了 `mamba_ssm` 就用 CUDA 内核，否则自动退回 `MambaTorch`（结果等价但更慢、更占显存）。`--mamba-impl cuda` 要求必须使用 CUDA 内核。
- `--tf32`（脚本默认开启）：允许 TF32 矩阵乘，4090 上更快，数值与 FP32 略有差异；需要严格 FP32 时，运行脚本前设 `EXTRA=""`。
- `--amp bf16`：混合精度。`MambaTorch` 内部的扫描始终以 FP32 计算。

## 3. 数据

```bash
bash scripts/download_data.sh ./data
```

从 Hugging Face 上的 Time-Series-Library 镜像下载 ETTh1/2、ETTm1/2、Weather、Electricity、Traffic、Exchange。PEMS03/04/07/08 不在该镜像中，需要自行把 `PEMS0X.npz` 放到 `./data/PEMS/`（iTransformer / S-Mamba 使用的版本）。

划分与官方一致：ETT 为 12/4/4 个月；Weather、Electricity、Traffic、Exchange 为 70/10/20，并且与 `Dataset_Custom` 一样把目标列 `OT` 移到最后（这会影响 S-Mamba 的扫描顺序）；PEMS 为 60/20/20。用训练段统计量标准化，日历特征与 `timeenc=1` 相同。

## 4. 运行

单次运行（结果追加到一个 JSONL 文件，每行包含全部配置和指标）：

```bash
python run.py --model smamba --data Electricity --pred-len 96 --identity add --seed 0 --tf32
python run.py --model itransformer --data Weather --pred-len 336 --identity lowrank --id-rank 8 --tf32
python run.py --model smamba --data Electricity --train-var-ratio 0.4 --identity add --id-dropout 0.3 --tf32
```

成批运行：

| 脚本 | 内容 | 运行次数 |
|---|---|---|
| `scripts/run_quick.sh` | ETTh1 / Weather / Electricity，预测 96 步，两种骨架 × {无, 有身份} × 3 个种子：先在官方规模上复核前期结论 | 36 |
| `scripts/run_main.sh` | 全部数据集 × 全部预测长度（PEMS 为 12/24/48/96，其余为 96/192/336/720）× 两种骨架 × {无, 有身份} × 3 个种子；缺少的数据集会跳过 | 全部数据齐时 576 |
| `scripts/run_ablation.sh` | S-Mamba，3 个数据集 × 3 个种子：满秩 / 低秩 r=8、32 / 逐层注入 / 身份 dropout / 10 倍身份学习率，以及未见变量协议（训练用 40% 的变量） | 90 |
| `scripts/run_order.sh` | 检验"S-Mamba 的固定扫描顺序本身起到了身份的作用"：S-Mamba 用随机扫描顺序训练 × {无, 有身份} × 3 个数据集 × 3 个种子，每次同时报告原始顺序和 4 个顺序平均的测试误差；固定顺序的对照直接复用 `run_quick.sh` 的结果（`RUN_FIXED=1` 可在此重跑） | 18 |

可以用环境变量缩小范围，例如 `DATASETS="Electricity Traffic" SEEDS="0" bash scripts/run_main.sh`。运行时间主要花在 Traffic、Electricity 和 PEMS07 上。

汇总与分析：

```bash
python tools/summarize.py results/quick.jsonl            # 均值 ± 标准差，逐种子配对比较与判定
python run.py --model smamba --data Electricity --identity add --save-ckpt checkpoints
python tools/analyze_identity.py checkpoints/<文件名>.pt --root-path ./data
```

`summarize.py` 的判定标准是事先固定的：每个种子上都比无身份版本好、且配对平均 MSE 改善至少 1%，才记为 effective；每个种子都更好但不足 1% 记为 marginal；每个种子都更差记为 worse；其余为 unclear。`analyze_identity.py` 检验学到的身份向量是否反映变量间的关系：计算身份余弦相似度与训练段 Pearson 相关的秩相关，以及"身份最近邻落在相关性最高的 10 个变量里"的比例。

## 5. 与官方实现的对应与差异

- **结构**：嵌入层（Linear(seq_len→d_model) + dropout，日历特征作为额外 token）、编码层（iTransformer：注意力 → 残差 → LN → 1×1 卷积 FFN → 残差 → LN；S-Mamba：前向 Mamba + 反向 Mamba 相加 → 残差 → LN → FFN → LN，Mamba 为 d_conv=2、expand=1）、最终 LN 与 Linear(d_model→pred_len)、实例归一化，都按官方代码实现。注意力使用 `scaled_dot_product_attention`，与官方的 FullAttention 数学上等价。
- **训练协议**：Adam、MSE、学习率从第 2 轮起每轮减半（`type1`）、验证 MSE 早停（patience 3）、取验证最优的权重测试；测试评估全部窗口，不丢弃任何 batch。
- **超参数**：`idembed/configs.py` 中每一项都标了来源。S-Mamba 在 ETTh1、Weather、Electricity 上的设置取自官方脚本；其余（包括所有 iTransformer 设置）标为 `assumed`，是按官方风格设定的，**尚未与官方脚本逐项核对**，正式报告数字前请先核对。
- **PEMS**：不加日历 token（官方代码会加一个全零的 token）。
- **随机性**：每次运行固定种子，但 CUDA 内核本身并非完全确定，同一种子的重复运行结果可能有微小差异。

## 6. 测试

```bash
python -m pytest -q tests        # CPU 上约 1 分钟
```

测试覆盖：
- 三种身份形式在初始化时与原模型逐位相同；
- 身份 dropout 只在训练时生效，日历 token 不受影响；
- 未见变量拿不到梯度、测试时为零身份；
- `MambaTorch` 的分块扫描与逐步递推一致，且满足因果性；
- 随机扫描顺序只影响训练和多顺序平均的测试，按原始顺序测试时与固定顺序模型完全相同；
- 官方划分边界、目标列后移和标准化；
- 命令行端到端运行，以及汇总、分析工具。

## 7. 当前状态

- **4090 上的快速实验已完成**（`run_quick.sh`，36 次运行）：两种骨架的基线都复现了论文数字（例如 S-Mamba 在 Electricity 上为 0.1389 / 0.2362，论文为 0.139 / 0.235；iTransformer 为 0.1482 / 0.2398，论文为 0.148 / 0.240）。身份嵌入在 Electricity 和 Weather 上对两种骨架都有效（3/3 个种子更好）：iTransformer 分别 −7.8%、−5.2%，S-Mamba 分别 −1.4%、−1.8%；在 ETTh1 上没有作用。原始结果和完整分析见 [newpaper 仓库](https://github.com/unforgettablerep/newpaper)。
- **CPU 上的验证**：21 项单元测试全部通过；用 S-Mamba 官方配置在 ETTh1 上完整训练，结果为 0.3866 / 0.4053，与论文一致。
- **结果文件现在记录实际使用的 Mamba 实现**（`mamba_backend`：`cuda` 为 mamba_ssm 内核，`torch` 为纯 PyTorch 回退，iTransformer 为 `null`）。快速实验的结果文件早于这项改动，没有这个字段。
- **加入了随机扫描顺序选项**，用来检验"S-Mamba 的提升比 iTransformer 小，是因为固定扫描顺序已经起到了一部分身份作用"这一推测（`scripts/run_order.sh`）。固定顺序时，加入该选项前后的计算逐位相同，所以新实验可以直接和已有的快速实验结果配对比较。
- **待运行**：`run_order.sh`、`run_ablation.sh`、`run_main.sh`。

## 8. 目录结构

```
idembed/          data.py（数据与划分） identity.py（身份嵌入） mamba.py（CUDA / 纯 PyTorch Mamba）
                  models.py（iTransformer、S-Mamba） engine.py（训练与评估） configs.py（超参数与来源）
run.py            单次运行的命令行入口
scripts/          download_data.sh  check_env.py  run_quick.sh  run_main.sh  run_ablation.sh  run_order.sh
tools/            summarize.py（结果汇总与判定）  analyze_identity.py（身份嵌入的可解释性）
tests/            test_core.py
experiments_cpu/  前期 CPU 探索实验（合成数据 Hub-SSM / 稀疏混合，真实数据上的改进对比），从 paperlib 迁入
```

## 9. 参考

- [S-Mamba：Is Mamba Effective for Time Series Forecasting?](https://arxiv.org/abs/2403.11144)、[iTransformer](https://arxiv.org/abs/2310.06625)、[Mamba](https://arxiv.org/abs/2312.00752)
- [STID](https://arxiv.org/abs/2208.05233)、[EMAformer](https://arxiv.org/abs/2511.08396)、[ss-Mamba](https://arxiv.org/abs/2506.14802)
- [Time-Series-Library 数据镜像](https://huggingface.co/datasets/thuml/Time-Series-Library)
