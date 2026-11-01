# RNK：可靠性门控的节点知识

newpaper 仓库 `papers/model_direction.md` 中模型的实现：在全局预测骨干上加三个模块。

| 模块 | 文件 | 内容 |
|---|---|---|
| 归纳式节点知识编码器（NKE） | `model.py` `KnowledgeEncoder` | 节点知识 = 描述信息编码 + 近期画像编码（过去 28 天按"星期 × 小时"的均值）+ 零初始化的残差身份。新节点没有身份，其余两项照常可用 |
| 知识条件的协变量一致分解（KCD） | `model.py` `CovariateResponse` | 协变量部分 $u$ = 共享基函数 × 由节点知识生成的系数 + 每类设备"每单位容量的响应" × 假设的容量。$u$ 先从回看窗口中扣除、预测后加回 |
| 台账与投运后验 | `registry.py` | 假设的容量按四种方式给出：登记日期即投运、时滞先验、投运后验（单变点，用晴天中午负荷对辐照的斜率）、真实日期（只用于合成数据） |
| 知识可靠性门控（KRG） | `gate.py`、`train.py` | $q$ 只作用于"依赖状态持续"的知识：身份乘以 $q$，28 天画像与 14 天短画像按 $q$ 混合；描述信息和 KCD 不受门控。$q$ 有两种来源：无标签的 $\chi^2$ 后验（4 个窗口统计量，阈值取 0.999 分位数），以及用前一天实测误差估计的定理 A1 最优权重（`gate_err`）。训练时以 0.2 的概率把 $q$ 置 0（持续性 dropout），使退路可用 |

所有量都以样本的负荷尺度 $\sigma$（过去 28 天的平均绝对负荷）为单位。全部模块关闭时就是普通的全局 MLP；初始化时，完整模型与普通骨干的输出完全相同（`tests/test_rnk.py`）。

训练细节：
- 节点知识经由零初始化的 FiLM 进入骨干。所以身份不能也从零开始，否则两个零因子相乘，梯度恒为 0。身份用标准差 0.02 的随机值初始化，FiLM 保持零初始化，初始化时输出仍与骨干相同。
- 训练时对权重做指数滑动平均（每步衰减 0.99），用平均后的权重做验证和早停。不做平均时，同一模型换种子在合成数据上的测试误差可相差 10% 以上，做平均后约为 0.5%。

## 运行

```bash
pip install torch numpy pandas scipy pyarrow holidays pytest
python -m pytest -q tests/test_rnk.py

# 合成数据：12 个变体 × 3 个种子，CPU 上每次约 1 分钟
OUT=results/rnk_sim.jsonl bash scripts/run_rnk_sim.sh
python tools/summarize_rnk.py results/rnk_sim.jsonl

# FeederBW（Zenodo 19550073；可以只用一个测量分包）
ZIPS="feeder_measurement_data_161-200.zip" WEATHER=weather_data.parquet META=feeder_metadata.csv \
  OUT=results/rnk_feederbw.jsonl bash scripts/run_rnk_feederbw.sh
python tools/summarize_rnk.py results/rnk_feederbw.jsonl --split temporal
```

## 变体与评估方式

变体见 `experiment.py`：`global`（无节点知识）、`id`、`id_drop`、`nke_noid`、`nke`、`kcd_reg`、`kcd_prior`、`kcd_post`、`kcd_oracle`、`rnk`（完整模型）、`rnk_noid`、`rnk_all`（门控全部知识的消融）。

每个训练好的模型在 `train.gate_modes` 的各种方式下评估：
- `trust`：始终用；
- `noid`：退路；
- `null`：不用任何知识；
- `gate`：无标签的 $\chi^2$ 门控；
- `gate_val`：在验证集上选阈值；
- `gate_err`：用前一天误差估计的权重；
- `gate_comb`：二者相乘；
- `oracle`：合成数据中已知变化时刻。

## 协议

**合成数据**（`sim.py`）：
- 120 个节点，其中 80 个用于训练、40 个是新节点；540 天；
- 结构：节点特有的周模式、光伏登记后平均 50 天才投运、测试期中 20% 的节点发生未记录的负荷转移。

**FeederBW**：
- 回看 168 小时，预测次日 24 小时；
- 训练期为 2023-04-29 至 2024-02-29，验证期为 2024 年 3 月，测试期为 2024-04-01 至 2025-03-31；
- 两种划分：按时间（全部馈线）、按地点（前一半馈线训练，后一半测试）。
