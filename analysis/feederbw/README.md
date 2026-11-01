# FeederBW 先导分析

newpaper `papers/application_direction.md` 第 3 节的数字由这些脚本算出，结果在 newpaper `results/feederbw_pilot/`。

数据来自 Zenodo 记录 [19550073](https://zenodo.org/records/19550073)（FeederBW v1.1，CC BY 4.0）：`feeder_metadata.csv`、`weather_data.parquet` 和测量数据分包 `feeder_measurement_data_161-200.zip`（馈线 161–200，1.67 GB）。

| 脚本 | 内容 | 用法 | 依赖 |
|---|---|---|---|
| `meta_stats.py` | 设备台账的统计：各类设备的装机变化、变更事件数与大小、测试年内的事件 | `python -I meta_stats.py feeder_metadata.csv` | pandas、numpy |
| `pilot_registry.py` | 全局 LightGBM 日前模型（小时级，3 个种子）：不用台账 / 馈线编号 / 冻结在第一行的台账 / 随时间变化的台账 / 后两者合用；按时间划分或按地点划分（前一半馈线训练、后一半测试） | `python -I pilot_registry.py --zip <分包> --weather weather_data.parquet --meta feeder_metadata.csv --out <目录> [--mode temporal\|locational]` | pandas、numpy、pyarrow、lightgbm |
| `pilot_event_timing.py` | 登记的光伏新增（≥ 10 kW）在量测中何时出现：晴天中午负荷对辐照的斜率在登记前后的变化，与登记容量对应的预期变化之比，并减去同期无光伏变更馈线的变化 | `python -I pilot_event_timing.py <目录>/hourly.pkl weather_data.parquet feeder_metadata.csv` | pandas、numpy、pyarrow |

`pilot_registry.py` 把小时聚合结果缓存为 `<目录>/hourly.pkl`，`pilot_event_timing.py` 读取它。
