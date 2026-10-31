# 分析脚本

newpaper 仓库 README 和 `papers/` 中的数字由这些脚本从结果文件算出，此前放在临时目录，这里保存以便复现。

- 读取结果文件的脚本都用 `python -I` 运行，路径通过参数传入。
- `<newpaper>` 为 newpaper 仓库目录，`<newpapercode>` 为本仓库目录。
- `traffic_deep.py`、`traffic_840.py` 需要 numpy，其余只用标准库（`off.py` 还会导入本仓库的 `idembed.configs` 和 `tools/summarize.py`）。

| 脚本 | 对应的 README 部分 | 用法 |
|---|---|---|
| `analyze_all.py` | 实验 2–4：主表与论文数字、两次消融、扫描顺序假设、重跑一致性 | `analyze_all.py <newpaper>/results/4090 [main\|ablation\|order\|repro\|all]` |
| `order_extra.py` | 实验 2：随机顺序 S-Mamba 与固定顺序、iTransformer 的对比 | `order_extra.py <newpaper>/results/4090` |
| `main_md.py` | 实验 4：主表（Markdown） | `main_md.py <newpaper>/results/4090/main.jsonl` |
| `traffic_deep.py`、`traffic_840.py` | 实验 5：Traffic 上身份损失集中在哪些传感器、传感器 840 的占比 | `traffic_deep.py <newpaper>/results/4090/traffic.jsonl` |
| `valsel.py` | 实验 5、7：在验证集上选择身份变体 | `valsel.py <newpaper>` |
| `round2.py` | 实验 6–8：列重排与 10 轮、身份学习率 ×10、全部数据集上的随机顺序 | `round2.py <newpaper> [perm\|idlr\|shuffle]` |
| `tables2.py` | 实验 7–8 的 Markdown 表 | `tables2.py <newpaper> [shuffle\|shuffle_agg\|idlr]` |
| `off.py` | 实验 9–10：官方设置视图（基线与论文、各变体、方案、验证集选择、综合对比表） | `off.py <newpaper> <newpapercode> [base\|identity\|select\|rules\|final\|final_md\|exp9\|exp10] [out.json]` |
| `budget.py` | `papers/theory.md` §5：身份参数的学习预算与收益 | `budget.py <newpaper>/results/4090` |

其他工具：`tools/summarize.py`（汇总表）、`tools/per_variable.py`（逐变量分析）、`tools/drift_gate_pilot.py`（漂移路由先导）、`experiments_sim/`（候选方向的沙盒模拟）。
