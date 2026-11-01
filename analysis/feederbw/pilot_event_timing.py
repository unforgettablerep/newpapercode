"""When does a registered PV addition become visible in the measured feeder load?

For every registry event with a PV addition of at least 10 kW, regress the sunny-midday hourly
load (10:00-15:59 local time) on global horizontal irradiance in a 60-day window before the
registry date and in windows 0-30, 30-60 and 60-120 days after it. The change of the slope is
compared with the change expected from the registered capacity, -0.8 * dPV / 1000 kW per W/m2
(performance ratio 0.8). The control-adjusted change subtracts the median slope change of the
feeders of the same pack that have no PV registry change from 60 days before to 120 days after the
event (this removes seasonal effects). A ratio of 1 means the full registered PV effect is visible.

Usage: python -I pilot_event_timing.py <out_dir>/hourly.pkl weather_data.parquet feeder_metadata.csv
(hourly.pkl is written by pilot_registry.py). Needs pandas, numpy, pyarrow.
"""
import sys

import numpy as np
import pandas as pd

WINDOWS = [(0, 30), (30, 60), (60, 120)]


def slope(x):
    if len(x) < 40 or x.ghi.std() < 50:
        return np.nan
    A = np.c_[np.ones(len(x)), x.ghi.values]
    coef, *_ = np.linalg.lstsq(A, x.P.values, rcond=None)
    return coef[1]


def slope_changes(f, t0):
    b0 = slope(f[(f.hour >= t0 - pd.Timedelta(days=60)) & (f.hour < t0)])
    return {f"{a}-{b}d": slope(f[(f.hour >= t0 + pd.Timedelta(days=a)) & (f.hour < t0 + pd.Timedelta(days=b))]) - b0
            for a, b in WINDOWS}


def main():
    hourly_path, weather_path, meta_path = sys.argv[1:4]
    h = pd.read_pickle(hourly_path)
    h["feeder"] = h["feeder"].astype("int64")
    h["hour"] = h["hour"].astype("datetime64[ns]")
    feeders = sorted(h.feeder.unique())

    w = pd.read_parquet(weather_path)
    w = w[w.feeder.isin(feeders)].copy()
    w["feeder"] = w["feeder"].astype("int64")
    w["hour"] = pd.to_datetime(w["timestamp_CET_CEST"], format="%Y-%m-%d %H:%M:%S").astype("datetime64[ns]")
    w["ghi"] = w["direct_radiation_W/m2"] + w["diffuse_radiation_W/m2"]
    w = w.groupby(["feeder", "hour"], as_index=False)["ghi"].mean()
    d = h.merge(w, on=["feeder", "hour"])
    d = d[d.hour.dt.hour.between(10, 15) & d.P.notna()]
    by_feeder = {k: v for k, v in d.groupby("feeder")}

    m = pd.read_csv(meta_path, parse_dates=["date"]).sort_values(["feeder", "date"])
    m = m[m.feeder.isin(feeders)].copy()
    m["dPV"] = m.groupby("feeder").PV_systems_kW.diff()
    events = m[m.dPV >= 10][["feeder", "date", "dPV"]]
    any_pv = m[m.dPV > 0][["feeder", "date"]]

    rows = []
    for _, e in events.iterrows():
        t0 = e.date
        expected = -0.8 * e.dPV / 1000
        own = slope_changes(by_feeder[e.feeder], t0)
        controls = []
        for fid, f in by_feeder.items():
            near = any_pv[(any_pv.feeder == fid) & (any_pv.date >= t0 - pd.Timedelta(days=60))
                          & (any_pv.date <= t0 + pd.Timedelta(days=120))]
            if fid != e.feeder and len(near) == 0:
                controls.append(slope_changes(f, t0))
        C = pd.DataFrame(controls)
        r = {"feeder": int(e.feeder), "date": t0.date(), "dPV_kW": e.dPV, "n_controls": len(C)}
        for k, v in own.items():
            r[f"ratio_{k}"] = v / expected
            r[f"ratio_adj_{k}"] = (v - C[k].median()) / expected
        rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(R.round(2).to_string(index=False))
    adj = R[[c for c in R.columns if c.startswith("ratio_adj")]]
    print("\nevents:", len(R), "| valid estimates per window:", adj.notna().sum().to_dict())
    print("\nmedian ratio of observed to expected slope change (1 = full registered PV effect visible):")
    print(R[[c for c in R.columns if c.startswith("ratio")]].median().round(2).to_string())
    print("\nquartiles of the control-adjusted ratio:")
    print(adj.quantile([0.25, 0.5, 0.75]).round(2).to_string())
    print("\nshare of events whose control-adjusted change reaches at least half of the expected change:")
    print((adj >= 0.5).sum().div(adj.notna().sum()).round(2).to_string())


if __name__ == "__main__":
    main()
