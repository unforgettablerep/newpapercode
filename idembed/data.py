"""Benchmark datasets with the splits, scaling and time features of the Time-Series-Library
(the code base of iTransformer and S-Mamba).

- ETTh*/ETTm*: 12/4/4 months for train/val/test.
- Weather, Electricity, Traffic, Exchange ("custom"): 70/10/20, target column `OT` moved last.
- PEMS03/04/07/08 (.npz, first feature): 60/20/20, no calendar features.
Values are standardised with training-split statistics (as sklearn's StandardScaler). Every
window of a split is served; nothing is dropped at evaluation time.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DATASETS = {   # name: (relative path, split rule, calendar frequency)
    "ETTh1": ("ETT-small/ETTh1.csv", "ett_hour", "h"),
    "ETTh2": ("ETT-small/ETTh2.csv", "ett_hour", "h"),
    "ETTm1": ("ETT-small/ETTm1.csv", "ett_minute", "t"),
    "ETTm2": ("ETT-small/ETTm2.csv", "ett_minute", "t"),
    "Weather": ("weather/weather.csv", "custom", "h"),
    "Electricity": ("electricity/electricity.csv", "custom", "h"),
    "Traffic": ("traffic/traffic.csv", "custom", "h"),
    "Exchange": ("exchange_rate/exchange_rate.csv", "custom", "h"),
    "PEMS03": ("PEMS/PEMS03.npz", "pems", None),
    "PEMS04": ("PEMS/PEMS04.npz", "pems", None),
    "PEMS07": ("PEMS/PEMS07.npz", "pems", None),
    "PEMS08": ("PEMS/PEMS08.npz", "pems", None),
}


def time_features(dates, freq):
    """timeenc=1 features of the Time-Series-Library, each scaled to [-0.5, 0.5]."""
    d = pd.DatetimeIndex(dates)
    feats = {
        "minute": d.minute / 59.0 - 0.5,
        "hour": d.hour / 23.0 - 0.5,
        "weekday": d.day_of_week / 6.0 - 0.5,
        "day": (d.day - 1) / 30.0 - 0.5,
        "yday": (d.day_of_year - 1) / 365.0 - 0.5,
    }
    names = {"h": ["hour", "weekday", "day", "yday"],
             "t": ["minute", "hour", "weekday", "day", "yday"],
             "d": ["weekday", "day", "yday"]}[freq]
    return np.stack([np.asarray(feats[n], dtype=np.float32) for n in names], 1)


def borders(rule, n, seq_len):
    if rule == "ett_hour":
        a, b, c = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
        return [0, a - seq_len, b - seq_len], [a, b, c]
    if rule == "ett_minute":
        a, b, c = 12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4
        return [0, a - seq_len, b - seq_len], [a, b, c]
    if rule == "custom":
        n_train, n_test = int(n * 0.7), int(n * 0.2)
        n_val = n - n_train - n_test
        return [0, n_train - seq_len, n - n_test - seq_len], [n_train, n_train + n_val, n]
    if rule == "pems":
        a, b = int(n * 0.6), int(n * 0.8)
        return [0, a - seq_len, b - seq_len], [a, b, n]
    raise ValueError(rule)


def read_raw(name, root):
    """Returns values (T, N) float32, calendar marks (T, M) float32 or None, variable names."""
    rel, rule, freq = DATASETS[name]
    path = Path(root) / rel
    if rule == "pems":
        with np.load(path, allow_pickle=False) as f:
            values = f["data"][:, :, 0].astype(np.float32)
        return values, None, [str(i) for i in range(values.shape[1])]
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "date"]
    if rule == "custom" and "OT" in cols:            # Dataset_Custom puts the target last
        cols = [c for c in cols if c != "OT"] + ["OT"]
    marks = time_features(pd.to_datetime(df["date"]), freq)
    return df[cols].values.astype(np.float32), marks, cols


@dataclass
class Split:
    values: torch.Tensor          # (T, N)
    marks: torch.Tensor | None    # (T, M)
    seq_len: int
    pred_len: int

    @property
    def n_windows(self):
        return len(self.values) - self.seq_len - self.pred_len + 1

    def batch(self, starts, var_idx=None):
        """starts: (B,) window starts. Returns x (B, L, n), x_mark (B, L, M) or None, y (B, H, n)."""
        t = starts[:, None] + torch.arange(self.seq_len + self.pred_len, device=starts.device)[None]
        v = self.values if var_idx is None else self.values[:, var_idx]
        w = v[t]
        x, y = w[:, :self.seq_len], w[:, self.seq_len:]
        m = None if self.marks is None else self.marks[t[:, :self.seq_len]]
        return x, m, y


def permutation(n, seed):
    """The fixed random order of the n variables used by load_splits(var_perm=seed)."""
    return np.random.default_rng(seed).permutation(n)


def load_splits(name, root, seq_len=96, pred_len=96, use_marks=True, device="cpu", var_perm=None):
    """var_perm: if given, the variables are listed in the fixed random order
    permutation(N, var_perm) instead of the file's column order."""
    values, marks, cols = read_raw(name, root)
    if var_perm is not None:
        perm = permutation(values.shape[1], var_perm)
        values, cols = values[:, perm], [cols[i] for i in perm]
    rule = DATASETS[name][1]
    b1, b2 = borders(rule, len(values), seq_len)
    train = values[b1[0]:b2[0]]
    mean, std = train.mean(0), train.std(0)
    std = np.where(std == 0, 1.0, std)
    values = (values - mean) / std
    v = torch.from_numpy(values).to(device)
    m = torch.from_numpy(marks).to(device) if (marks is not None and use_marks) else None
    splits = [Split(v[s:e], None if m is None else m[s:e], seq_len, pred_len) for s, e in zip(b1, b2)]
    return splits, cols
