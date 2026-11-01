"""Pilot: does a time-varying asset registry help a global day-ahead model of LV feeder load?

Data: one FeederBW measurement pack (40 feeders), the FeederBW weather file and the feeder
metadata (registry). Hourly active power, day-ahead forecasts issued at 00:00, global LightGBM
over all feeders of the pack, 3 seeds.

Variants: V0 no feeder information; V1 feeder id (transductive); V2 registry frozen at the first
metadata row (stale); V3 time-varying registry; V4 time-varying registry + feeder id.

Splits:
  temporal   train 2023-04..2024-02, validation 2024-03, test 2024-04..2025-03 (same feeders)
  locational the same periods, but train/validation on the first half of the feeders of the pack
             and test on the other half (unseen feeders; variants with feeder id are skipped)

Usage:
  python -I pilot_registry.py --zip feeder_measurement_data_161-200.zip --weather weather_data.parquet \
      --meta feeder_metadata.csv --out <dir> [--mode temporal|locational]
Needs pandas, numpy, pyarrow, lightgbm. The hourly aggregation is cached in <dir>/hourly.pkl.
"""
import argparse
import io
import json
import os
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TRAIN_END = pd.Timestamp("2024-03-01")
VAL_END = pd.Timestamp("2024-04-01")
TEST_END = pd.Timestamp("2025-04-01")
REG_COLS = ["PV_systems_kW", "batteries_kW", "heat_pumps_kW", "EV_chargers_kW", "housing_units_count",
            "industry_and_commerce_count", "storage_heaters_kW"]


def load_hourly(zip_path, cache):
    """Hourly mean active power per feeder in local time; hours with < 30 one-minute values are missing."""
    if os.path.exists(cache):
        hourly = pd.read_pickle(cache)
    else:
        frames = []
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".parquet"):
                    continue
                with z.open(name) as fh:
                    table = pq.read_table(io.BytesIO(fh.read()),
                                          columns=["feeder", "timestamp_CET_CEST", "active_power_kW"])
                df = table.to_pandas()
                df.columns = ["feeder", "ts", "P"]
                frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        raw["ts"] = pd.to_datetime(raw["ts"], format="%Y-%m-%d %H:%M:%S")
        raw["hour"] = raw["ts"].dt.floor("h")
        hourly = raw.groupby(["feeder", "hour"], as_index=False)["P"].mean()
        count = raw.groupby(["feeder", "hour"]).size().rename("n").reset_index()
        hourly = hourly.merge(count, on=["feeder", "hour"])
        hourly.loc[hourly.n < 30, "P"] = np.nan
        hourly.to_pickle(cache)
    hourly["feeder"] = hourly["feeder"].astype("int64")
    hourly["hour"] = hourly["hour"].astype("datetime64[ns]")
    return hourly


def build_frame(hourly, weather_path, meta_path):
    feeders = sorted(hourly.feeder.unique())
    grid = pd.MultiIndex.from_product(
        [feeders, pd.date_range("2023-04-01", "2025-03-31 23:00", freq="h")], names=["feeder", "hour"])
    H = hourly.set_index(["feeder", "hour"])["P"].reindex(grid).reset_index()
    H["feeder"] = H["feeder"].astype("int64")
    H["hour"] = H["hour"].astype("datetime64[ns]")

    w = pd.read_parquet(weather_path)
    w = w[w.feeder.isin(feeders)].copy()
    w["feeder"] = w["feeder"].astype("int64")
    w["hour"] = pd.to_datetime(w["timestamp_CET_CEST"], format="%Y-%m-%d %H:%M:%S")
    w["ghi"] = w["direct_radiation_W/m2"] + w["diffuse_radiation_W/m2"]
    w = w.rename(columns={"air_temperature_C": "temp"})[["feeder", "hour", "ghi", "temp"]]
    w = w.groupby(["feeder", "hour"], as_index=False).mean()          # DST duplicates
    w["hour"] = w["hour"].astype("datetime64[ns]")
    H = H.merge(w, on=["feeder", "hour"], how="left")

    m = pd.read_csv(meta_path, parse_dates=["date"]).sort_values(["feeder", "date"])
    m = m[m.feeder.isin(feeders)].copy()
    m["feeder"] = m["feeder"].astype("int64")
    m["date"] = m["date"].astype("datetime64[ns]")
    # each metadata row is valid from its date until the next row of the same feeder
    H["day"] = H["hour"].dt.normalize().astype("datetime64[ns]")
    H = pd.merge_asof(H.sort_values("day"), m[["feeder", "date"] + REG_COLS].sort_values("date"),
                      left_on="day", right_on="date", by="feeder", direction="backward").drop(columns="date")
    first = m.groupby("feeder")[REG_COLS].first()
    for c in REG_COLS:
        H[c + "_static"] = H.feeder.map(first[c])

    # last large registry change (PV >= 10 kW, heat pump >= 5 kW, EV charger >= 11 kW) before each day
    for c in ["PV_systems_kW", "heat_pumps_kW", "EV_chargers_kW"]:
        m["d_" + c] = m.groupby("feeder")[c].diff().fillna(0)
    big = (m["d_PV_systems_kW"] >= 10) | (m["d_heat_pumps_kW"] >= 5) | (m["d_EV_chargers_kW"] >= 11)
    ev = m[big][["feeder", "date", "d_PV_systems_kW", "d_heat_pumps_kW", "d_EV_chargers_kW"]]
    H = pd.merge_asof(H.sort_values("day"), ev.rename(columns={"date": "ev_date"}).sort_values("ev_date"),
                      left_on="day", right_on="ev_date", by="feeder", direction="backward")
    H["days_since_event"] = (H["day"] - H["ev_date"]).dt.days
    H = H.sort_values(["feeder", "hour"]).reset_index(drop=True)

    # day-ahead features: the latest value used for any target hour is from the previous day
    g = H.groupby("feeder")["P"]
    H["lag24"] = g.shift(24)
    H["lag48"] = g.shift(48)
    H["lag168"] = g.shift(168)
    H["mean_prev_day"] = g.transform(lambda s: s.shift(24).rolling(24, min_periods=12).mean())
    H["mean_prev_week_samehour"] = sum(g.shift(24 * k) for k in range(1, 8)) / 7
    H["ghi_lag24"] = H.groupby("feeder")["ghi"].shift(24)
    H["hod"] = H["hour"].dt.hour
    H["dow"] = H["hour"].dt.dayofweek
    H["doy"] = H["hour"].dt.dayofyear
    H["pv_x_ghi"] = H["PV_systems_kW"] * H["ghi"] / 1000.0
    H["pv_x_ghi_static"] = H["PV_systems_kW_static"] * H["ghi"] / 1000.0
    H["pv_x_ghi_lag24"] = H["PV_systems_kW"] * H["ghi_lag24"] / 1000.0
    H["hp_x_cold"] = H["heat_pumps_kW"] * np.clip(15 - H["temp"], 0, None)
    H["hp_x_cold_static"] = H["heat_pumps_kW_static"] * np.clip(15 - H["temp"], 0, None)
    return H, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="temporal", choices=["temporal", "locational"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    hourly = load_hourly(args.zip, os.path.join(args.out, "hourly.pkl"))
    H, m = build_frame(hourly, args.weather, args.meta)

    base = ["lag24", "lag48", "lag168", "mean_prev_day", "mean_prev_week_samehour", "ghi", "ghi_lag24", "temp",
            "hod", "dow", "doy"]
    static = [c + "_static" for c in REG_COLS] + ["pv_x_ghi_static", "hp_x_cold_static"]
    dynamic = REG_COLS + ["pv_x_ghi", "pv_x_ghi_lag24", "hp_x_cold"]
    variants = {
        "V0_global_no_registry": base,
        "V1_feeder_id": base + ["feeder"],
        "V2_registry_static": base + static,
        "V3_registry_time_varying": base + dynamic,
        "V4_time_varying_plus_id": base + dynamic + ["feeder"],
    }

    D = H.dropna(subset=["P", "lag24", "lag168", "ghi"])
    tr = D[D.hour < TRAIN_END]
    va = D[(D.hour >= TRAIN_END) & (D.hour < VAL_END)]
    te = D[(D.hour >= VAL_END) & (D.hour < TEST_END)].copy()
    if args.mode == "locational":
        feeders = sorted(D.feeder.unique())
        seen = set(feeders[: len(feeders) // 2])
        tr = tr[tr.feeder.isin(seen)]
        va = va[va.feeder.isin(seen)]
        te = te[~te.feeder.isin(seen)].copy()
        variants = {k: v for k, v in variants.items() if "feeder" not in v}
    print("mode", args.mode, "| rows train/val/test", len(tr), len(va), len(te),
          "| test feeders", te.feeder.nunique())

    params = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
    out = {"mode": args.mode, "overall": {}}
    for name, feats in variants.items():
        preds = []
        for seed in (0, 1, 2):
            cat = ["feeder"] if "feeder" in feats else []
            dtr = lgb.Dataset(tr[feats], tr["P"], categorical_feature=cat, free_raw_data=False)
            dva = lgb.Dataset(va[feats], va["P"], categorical_feature=cat, reference=dtr)
            booster = lgb.train(dict(params, seed=seed), dtr, num_boost_round=3000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            preds.append(booster.predict(te[feats], num_iteration=booster.best_iteration))
        te[name] = np.mean(preds, axis=0)
        e = te[name] - te["P"]
        out["overall"][name] = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}
        print(f"{name:28s} MAE {out['overall'][name]['MAE']:.3f}  RMSE {out['overall'][name]['RMSE']:.3f}")

    cols = list(variants)
    sunny_mid = te.hod.between(10, 15) & (te.ghi > 400)
    post = te.days_since_event.between(0, 30)
    masks = {
        "sunny_midday": sunny_mid,
        "0-30d_after_registered_change": post,
        "0-30d_after_PV>=10kW_sunny_midday": post & (te.d_PV_systems_kW >= 10) & sunny_mid,
        "no_change_in_last_90d": ~te.days_since_event.between(0, 90),
    }
    for label, mask in masks.items():
        out[label] = {}
        print(f"\n== {label} (rows {int(mask.sum())})")
        for c in cols:
            e = te.loc[mask, c] - te.loc[mask, "P"]
            out[label][c] = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                             "n": int(mask.sum())}
            print(f"  {c:28s} MAE {out[label][c]['MAE']:.3f}  RMSE {out[label][c]['RMSE']:.3f}")

    per_feeder = te.groupby("feeder").apply(
        lambda d: pd.Series({c: np.mean((d[c] - d["P"]) ** 2) for c in cols}), include_groups=False)
    v0 = per_feeder["V0_global_no_registry"].sort_values(ascending=False)
    share = v0.cumsum() / v0.sum()
    out["mse_share_top1_top4_feeders_V0"] = [float(share.iloc[0]), float(share.iloc[3])]
    out["top4_feeders_V0"] = [int(f) for f in share.index[:4]]
    gain = per_feeder["V3_registry_time_varying"] / per_feeder["V0_global_no_registry"] - 1
    out["per_feeder_mse_change_V3_vs_V0"] = {"median": float(gain.median()), "share_improved": float((gain < 0).mean())}
    print("\nMSE share of the top-1 / top-4 feeders under V0: %.3f / %.3f, feeders %s"
          % (share.iloc[0], share.iloc[3], out["top4_feeders_V0"]))
    print("per-feeder MSE change V3 vs V0: median %.3f, share improved %.2f" % (gain.median(), (gain < 0).mean()))
    path = os.path.join(args.out, f"pilot_registry_{args.mode}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
