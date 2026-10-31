"""Markdown tables from the JSON files written by sim_directions.py.

python experiments_sim/tables.py <dir with sim_A.json, sim_A2.json, sim_B.json, sim_D.json, sim_E.json>
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
load = lambda n: json.load(open(D / f"sim_{n}.json"))
pct = lambda a, b: f"{(a / b - 1) * 100:+.2f}%".replace("-", "−")


def table_A(rows, title, key_cols):
    print(f"\n**{title}**（测试 MSE；括号内为相对最好的单一模型）\n")
    print("| " + " | ".join(key_cols) + " | 局部相对全局 | 全局 | 局部 | 静态 BG | 后验（固定阈值） | 后验（验证集选阈值） | 硬门控 | 状态已知（上限） | 后验占上限收益 | 检测 AUC |")
    print("|" + "---|" * (len(key_cols) + 10))
    for k, v in rows.items():
        kv = dict(part.split("=") for part in k.split())
        g, l, s, pf, pv, h, o = (v[m]["all"] for m in ("global", "local", "static_bg", "posterior_fixed",
                                                         "posterior_val", "hard_val", "oracle_regime"))
        best = min(g, l)
        share = f"{100 * (s - pf) / (s - o):.0f}%" if (s - o) / s > 1e-3 else "–"   # gap < 0.1 %: n/a
        a = v.get("auc_D2")
        print("| " + " | ".join(kv[c] for c in key_cols) + f" | {pct(l, g)} | {g:.4f} | {l:.4f} | {s:.4f} | "
              f"**{pf:.4f}**（{pct(pf, best)}） | {pv:.4f} | {h:.4f} | {o:.4f} | {share} | "
              f"{'–' if a is None else f'{a:.2f}'} |")


A = load("A")
table_A(A["A"], "A：只有日周期（变量特有结构在 48 小时窗口中可见）", ["tau", "rho", "drift", "timing"])
table_A(A["A_weekly"], "A：加入变量特有的周周期（168 小时，窗口中不可见）", ["weekly", "rho", "timing"])

A2 = load("A2")["A2"]
print("\n**A2：跨变量消息**（测试 MSE：稳定变量 / 全部变量）\n")
print("| 漂移比例 | 通道独立 | 跨变量（均匀） | 跨变量（可靠性加权） | 跨变量（状态已知） | 可靠性加权 + 自身输出门控 |")
print("|---|---|---|---|---|---|")
for k, v in A2.items():
    c = lambda m: f"{v[m]['stable_vars']:.4f} / {v[m]['all']:.4f}"
    print(f"| {k.split('=')[1]} | {c('channel_independent')} | {c('cross_uniform')} | {c('cross_reliability')} | "
          f"{c('cross_oracle')} | **{c('cross_reliability+self_gate')}** |")

B = load("B")["B"]
print("\n**B：区间预测**（目标覆盖率 0.90；覆盖率：全部 / 漂移窗口 / 稳定变量；平均宽度）\n")
print("| 设定 | 校准集中高 D² 窗口数 | 边际 | 逐变量 | Mondrian（按 D² 分组） |")
print("|---|---|---|---|---|")
for k, v in B.items():
    c = lambda m: (f"{v[m]['coverage_all']:.3f} / {v[m]['coverage_drifted']:.3f} / "
                   f"{v[m]['coverage_stable_vars']:.3f}；{v[m]['width']:.3f}")
    print(f"| {k} | {v['n_cal_high_D2']:.0f} | {c('marginal')} | {c('per_variable')} | {c('mondrian_D2')} |")

Dd = load("D")["D"]
print("\n**D：集成与权重平均**（测试 MSE，5 个模型）\n")
print("| 单模型 | 输出集成 | 从头训练后权重平均 | EMA（衰减 0.99） | SWA（单次运行） | 同起点：单模型 | 同起点：输出集成 | 同起点：权重平均（模型汤） | 模型间方差 V | 实际集成收益 |")
print("|---|---|---|---|---|---|---|---|---|---|")
print(f"| {Dd['single']:.4f} | {Dd['output_ensemble']:.4f} | {Dd['weight_avg_from_scratch']:.4f} | {Dd['ema']:.4f} | "
      f"{Dd['swa_one_run']:.4f} | {Dd['shared_start_single']:.4f} | {Dd['shared_start_output_ensemble']:.4f} | "
      f"{Dd['shared_start_soup']:.4f} | {Dd['seed_variance']:.4f} | {Dd['actual_ensemble_gain']:.4f} |")

E = load("E")["E"]
print("\n**E：变量级稳健训练**（一个噪声很大的变量主导损失；测试 MSE）\n")
print("| 目标 | 平均 MSE | 去掉该变量后的平均 MSE | 变量 MSE 中位数 |")
print("|---|---|---|---|")
for k, v in E.items():
    print(f"| {k} | {v['avg_mse']:.4f} | {v['avg_mse_without_heavy']:.4f} | {v['median_var_mse']:.4f} |")
