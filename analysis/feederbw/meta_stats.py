import sys
import pandas as pd
import numpy as np

path = sys.argv[1]
m = pd.read_csv(path, parse_dates=["date"])
m = m.sort_values(["feeder", "date"])
print("rows", len(m), "feeders", m.feeder.nunique(), "substations", m.substation.nunique())
print("date range", m.date.min().date(), m.date.max().date())
rows = m.groupby("feeder").size()
print("rows per feeder: mean %.1f median %d min %d max %d" % (rows.mean(), rows.median(), rows.min(), rows.max()))

cap = ["PV_systems_kW", "heat_pumps_kW", "EV_chargers_kW", "batteries_kW",
       "storage_heaters_kW", "electric_heaters_kW", "hot_water_tanks_kW",
       "flow-type_heaters_kW", "other_consumers_kW", "chp_kW"]
cnt = ["housing_units_count", "industry_and_commerce_count"]
first = m.groupby("feeder").first()
last = m.groupby("feeder").last()

print("\n== totals over all feeders: first row -> last row ==")
for c in cap + cnt:
    a, b = first[c].sum(), last[c].sum()
    print(f"{c:24s} {a:10.1f} -> {b:10.1f}  ({(b-a)/a*100 if a else float('nan'):+.1f}%)  feeders with change: {(last[c]!=first[c]).sum()}")

# events: row-to-row changes
d = m.groupby("feeder")[cap + cnt].diff()
m2 = m.copy()
for c in cap + cnt:
    m2["d_" + c] = d[c]
ev = m2.dropna(subset=["d_PV_systems_kW"])
print("\nchange rows (excluding first row per feeder):", len(ev))
for c in ["PV_systems_kW", "heat_pumps_kW", "EV_chargers_kW", "batteries_kW"]:
    x = ev["d_" + c]
    nz = x[x != 0]
    print(f"{c:18s} events {len(nz):4d} (pos {int((nz>0).sum())}, neg {int((nz<0).sum())}); "
          f"|delta| kW median {nz.abs().median() if len(nz) else 0:.1f}, p90 {nz.abs().quantile(.9) if len(nz) else 0:.1f}, max {nz.abs().max() if len(nz) else 0:.1f}")

# test-year (2024-04-01 .. 2025-03-31) events
test = ev[(ev.date >= "2024-04-01") & (ev.date <= "2025-03-31")]
print("\n== events inside the recommended test year (2024-04-01..2025-03-31) ==")
for c in ["PV_systems_kW", "heat_pumps_kW", "EV_chargers_kW", "batteries_kW"]:
    x = test["d_" + c]; nz = x[x != 0]
    print(f"{c:18s} events {len(nz):4d}, feeders {test.loc[x != 0, 'feeder'].nunique():3d}, total delta {nz.sum():8.1f} kW")

# relative size: change of PV / HP / EV over the 2 years relative to a crude feeder size proxy
size = first["housing_units_count"] + first["industry_and_commerce_count"]
rel = pd.DataFrame({
    "PV_add_kW": last["PV_systems_kW"] - first["PV_systems_kW"],
    "HP_add_kW": last["heat_pumps_kW"] - first["heat_pumps_kW"],
    "EV_add_kW": last["EV_chargers_kW"] - first["EV_chargers_kW"],
    "BAT_add_kW": last["batteries_kW"] - first["batteries_kW"],
    "PV_first_kW": first["PV_systems_kW"],
    "units": size,
})
rel["PV_add_per_unit"] = rel.PV_add_kW / rel.units.clip(lower=1)
rel["PV_growth_rel"] = rel.PV_add_kW / rel.PV_first_kW.replace(0, np.nan)
print("\n== per-feeder change over 2 years ==")
print(rel[["PV_add_kW", "HP_add_kW", "EV_add_kW", "BAT_add_kW", "PV_add_per_unit"]].describe(percentiles=[.5, .75, .9, .95]).round(2).to_string())
print("\nfeeders with PV added >= 30 kW:", int((rel.PV_add_kW >= 30).sum()), "| >= 100 kW:", int((rel.PV_add_kW >= 100).sum()))
print("feeders whose PV more than doubled (from >0):", int((rel.PV_growth_rel > 1).sum()), "| PV from 0 to >0:", int(((rel.PV_first_kW == 0) & (rel.PV_add_kW > 0)).sum()))
print("\nlargest PV additions (feeder, kW, first kW, units):")
print(rel.sort_values("PV_add_kW", ascending=False).head(10)[["PV_add_kW", "PV_first_kW", "units"]].to_string())
print("\nfeeder 63 rows:")
print(m[m.feeder == 63][["date", "PV_systems_kW", "batteries_kW", "heat_pumps_kW", "EV_chargers_kW", "housing_units_count"]].to_string(index=False))
