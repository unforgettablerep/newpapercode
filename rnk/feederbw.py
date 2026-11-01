"""FeederBW (Treutlein et al., Scientific Data 2026; Zenodo 19550073) as an RNK dataset dict.

Hourly mean active power per feeder in local time (hours with < 30 one-minute values are
missing), from 2023-04-01 00:00 to 2025-03-31 23:00 on a naive local-time grid (the spring DST
hour is missing, the autumn one is merged). Covariates: global irradiance (direct + diffuse,
kW/m2) and (temperature - 10) / 10 from the dataset's weather file (ex-post values).
Calendar: hour and weekday encodings and a weekend / public-holiday flag (Baden-Wuerttemberg).

Registry (feeder_metadata.csv): each row holds from its date to the next row. Technologies:
PV, batteries, heat pumps, EV chargers, storage heaters; cap0 is the first row, every change
between consecutive rows is an event at that row's date. Static descriptors (first row):
log(1 + housing units), log(1 + industry and commerce units), log(1 + annual kWh) of the seven
commerce profiles.
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

KINDS = ("PV_systems_kW", "batteries_kW", "heat_pumps_kW", "EV_chargers_kW", "storage_heaters_kW")
SIGNS = ("mono", "free", "mono", "mono", "mono")
STATIC = ("housing_units_count", "industry_and_commerce_count", "g0_general_kWh", "g1_workdays_kWh",
          "g2_evening_kWh", "g3_continuous_kWh", "g4_shop_hairdresser_kWh", "g5_bakery_kWh",
          "g6_weekend_kWh")
START, END = pd.Timestamp("2023-04-01 00:00"), pd.Timestamp("2025-03-31 23:00")


def load_hourly(zip_paths, cache):
    if os.path.exists(cache):
        h = pd.read_pickle(cache)
    else:
        import pyarrow.parquet as pq
        frames = []
        for zp in zip_paths:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if name.endswith(".parquet"):
                        with z.open(name) as fh:
                            t = pq.read_table(io.BytesIO(fh.read()),
                                              columns=["feeder", "timestamp_CET_CEST", "active_power_kW"])
                        df = t.to_pandas()
                        df.columns = ["feeder", "ts", "P"]
                        frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        raw["hour"] = pd.to_datetime(raw["ts"], format="%Y-%m-%d %H:%M:%S").dt.floor("h")
        h = raw.groupby(["feeder", "hour"], as_index=False)["P"].mean()
        n = raw.groupby(["feeder", "hour"]).size().rename("n").reset_index()
        h = h.merge(n, on=["feeder", "hour"])
        h.loc[h.n < 30, "P"] = np.nan
        h.to_pickle(cache)
    h["feeder"] = h["feeder"].astype("int64")
    h["hour"] = h["hour"].astype("datetime64[ns]")
    return h


def make_feederbw(hourly, weather_path, meta_path, feeders=None):
    import holidays
    feeders = sorted(hourly.feeder.unique()) if feeders is None else list(feeders)
    grid = pd.date_range(START, END, freq="h")
    T, N = len(grid), len(feeders)
    pos = {f: i for i, f in enumerate(feeders)}
    y = np.full((N, T), np.nan)
    hh = hourly[hourly.feeder.isin(feeders)]
    ti = ((hh.hour - START) / pd.Timedelta(hours=1)).astype(int).values
    ok = (ti >= 0) & (ti < T)
    y[hh.feeder.map(pos).values[ok], ti[ok]] = hh.P.values[ok]

    w = pd.read_parquet(weather_path, columns=["feeder", "timestamp_CET_CEST", "direct_radiation_W/m2",
                                               "diffuse_radiation_W/m2", "air_temperature_C"])
    w = w[w.feeder.isin(feeders)].copy()
    w["hour"] = pd.to_datetime(w["timestamp_CET_CEST"], format="%Y-%m-%d %H:%M:%S")
    w["irr"] = (w["direct_radiation_W/m2"] + w["diffuse_radiation_W/m2"]) / 1000.0
    w["temp"] = (w["air_temperature_C"] - 10.0) / 10.0
    w = w.groupby(["feeder", "hour"], as_index=False)[["irr", "temp"]].mean()
    cov = np.full((N, T, 2), np.nan)
    wi = ((w.hour - START) / pd.Timedelta(hours=1)).astype(int).values
    ok = (wi >= 0) & (wi < T)
    fi = w.feeder.map(pos).values
    cov[fi[ok], wi[ok], 0] = w.irr.values[ok]
    cov[fi[ok], wi[ok], 1] = w.temp.values[ok]
    for c in range(2):                                    # gaps (spring DST hour): carry forward
        cov[:, :, c] = pd.DataFrame(cov[:, :, c].T).ffill().bfill().fillna(0.0).values.T

    hod = grid.hour.values
    dow = grid.dayofweek.values
    hol = holidays.Germany(subdiv="BW", years=[2023, 2024, 2025])
    off = np.array([d in hol for d in grid.date]) | (dow >= 5)
    cal = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), off.astype(float)], 1)

    m = pd.read_csv(meta_path, parse_dates=["date"]).sort_values(["feeder", "date"])
    m = m[m.feeder.isin(feeders)]
    cap0 = np.zeros((N, len(KINDS)))
    static = np.zeros((N, len(STATIC)))
    events = []
    for f, g in m.groupby("feeder"):
        i = pos[f]
        g = g.reset_index(drop=True)
        cap0[i] = g.loc[0, list(KINDS)].values.astype(float)
        static[i] = np.log1p(g.loc[0, list(STATIC)].values.astype(float))
        for r in range(1, len(g)):
            d = int((g.loc[r, "date"] - START) / pd.Timedelta(days=1))
            for k, col in enumerate(KINDS):
                delta = float(g.loc[r, col] - g.loc[r - 1, col])
                if delta != 0 and 0 <= d < T // 24:
                    events.append({"node": i, "kind": k, "tau": 24 * d, "delta": delta})
    return {
        "y": y, "cov": cov, "cal": cal, "hod": hod, "dow": dow, "cap0": cap0, "events": events,
        "static": static, "kinds": KINDS, "signs": SIGNS, "pv_kind": 0, "irr_col": 0,
        "feeders": np.array(feeders), "days": T // 24,
    }
