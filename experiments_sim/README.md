# 沙盒模拟：候选方向的可行性

检验 newpaper `papers/theory_directions.md` 中各方向的推导：用结构已知的合成多变量序列，配合线性和小型神经网络预测器，只用 CPU。

```bash
python experiments_sim/sim_directions.py all --seeds 3 --out results.json   # 或 A / A2 / B / D / E
python experiments_sim/tables.py <结果目录>                                    # 生成 Markdown 表格
```

| 部分 | 检验的推导 |
|---|---|
| A | 定理 A1 / 推论 A1：动态（后验 × Bates–Granger）组合的收益 = 局部知识的价值 × 检测器的信息量；推论 A2：阈值按误报率设定 |
| A2 | 跨变量消息的可靠性加权；与自身输出门控联合使用 |
| B | Mondrian 共形预测在漂移组上的覆盖率；边际区间的覆盖不足 |
| D | 集成收益恒等于模型间方差；权重平均只在共同起点下成立 |
| E | 重新加权在同分布平均 MSE 下不会优于 ERM |

运行时间：在 4 线程 CPU 上约 20 分钟（A 部分最长）。
