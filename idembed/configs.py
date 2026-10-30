"""Default hyperparameters per (backbone, dataset, horizon).

"official" (the default) copies the run scripts of the official repositories, horizon by
horizon (S-Mamba: wzhwzhwzh0921/S-D-Mamba, scripts/multivariate_forecasting; iTransformer:
thuml/iTransformer, same path). Arguments a script leaves unset take the defaults of that
repository's run.py: learning rate 1e-4, batch size 32, 10 epochs, and for S-Mamba
d_state 32. `source` records where a setting comes from:
- "S-Mamba script" / "iTransformer script": checked against the official script;
- "assumed": not in the official scripts we checked (PEMS); compare before reporting.

"legacy" is the table used up to newpapercode a48f354 (one setting per dataset, partly
assumed); keep it to reproduce or extend those results. Every value can be overridden from
the command line.
"""

COMMON = dict(seq_len=96, dropout=0.1, activation="gelu", n_heads=8, use_norm=True,
              patience=3, batch_size=32, epochs=10)

_S, _I = "S-Mamba script", "iTransformer script"

# official settings at horizon 96, and the values that differ at the other horizons
SMAMBA = {
    "ETTh1": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=7e-5, source=_S),
    "ETTh2": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=4e-5, source=_S),
    "ETTm1": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=5e-5, source=_S),
    "ETTm2": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=5e-5, source=_S),
    "Exchange": dict(e_layers=2, d_model=128, d_ff=128, d_state=32, lr=1e-4, batch_size=16,
                     source=_S),
    "Weather": dict(e_layers=3, d_model=512, d_ff=512, d_state=2, lr=5e-5, epochs=5, source=_S),
    "Electricity": dict(e_layers=3, d_model=512, d_ff=512, d_state=16, lr=1e-3, epochs=5,
                        batch_size=16, source=_S),
    "Traffic": dict(e_layers=4, d_model=512, d_ff=512, d_state=32, lr=1e-3, batch_size=16,
                    source=_S),
    "PEMS03": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
    "PEMS04": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
    "PEMS07": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
    "PEMS08": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
}
_small = dict(d_model=128, d_ff=128)
SMAMBA_H = {
    "ETTh1": {336: dict(lr=5e-5), 720: dict(lr=5e-5)},
    "ETTh2": {336: dict(lr=3e-5), 720: dict(lr=7e-5)},
    "ETTm1": {192: _small, 336: _small, 720: _small},
    "ETTm2": {192: _small, 336: dict(_small, lr=3e-5), 720: _small},
    "Exchange": {192: dict(batch_size=32), 336: dict(batch_size=32, lr=5e-5),
                 720: dict(batch_size=32, lr=5e-5)},
    "Electricity": {H: dict(d_state=32, lr=5e-4) for H in (192, 336, 720)},
    "Traffic": {336: dict(lr=2e-3), 720: dict(lr=8e-4)},
}

ITRANSFORMER = {
    "ETTh1": dict(e_layers=2, d_model=256, d_ff=256, lr=1e-4, source=_I),
    "ETTh2": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source=_I),
    "ETTm1": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source=_I),
    "ETTm2": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source=_I),
    "Weather": dict(e_layers=3, d_model=512, d_ff=512, lr=1e-4, source=_I),
    "Electricity": dict(e_layers=3, d_model=512, d_ff=512, lr=5e-4, batch_size=16, source=_I),
    "Traffic": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, batch_size=16, source=_I),
    "Exchange": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source=_I),
    "PEMS03": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
    "PEMS04": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
    "PEMS07": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
    "PEMS08": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
}
ITRANSFORMER_H = {
    "ETTh1": {336: dict(d_model=512, d_ff=512), 720: dict(d_model=512, d_ff=512)},
    "Exchange": {336: dict(epochs=1)},              # as in the official script
}

LEGACY = {
    "smamba": {
        "ETTh1": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=7e-5,
                      source="S-Mamba script"),
        "Weather": dict(e_layers=3, d_model=512, d_ff=512, d_state=2, lr=5e-5, epochs=5,
                        source="S-Mamba script"),
        "Electricity": dict(e_layers=3, d_model=512, d_ff=512, d_state=16, lr=1e-3, epochs=5,
                            batch_size=16, source="S-Mamba script"),
        "ETTh2": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=7e-5, source="assumed"),
        "ETTm1": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=5e-5, source="assumed"),
        "ETTm2": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=5e-5, source="assumed"),
        "Exchange": dict(e_layers=2, d_model=128, d_ff=128, d_state=2, lr=1e-4, source="assumed"),
        "Traffic": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, epochs=5,
                        batch_size=16, source="assumed"),
    },
    "itransformer": {
        "ETTh1": dict(e_layers=2, d_model=256, d_ff=256, lr=1e-4, source="assumed"),
        "ETTh2": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
        "ETTm1": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
        "ETTm2": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
        "Weather": dict(e_layers=3, d_model=512, d_ff=512, lr=1e-4, source="assumed"),
        "Electricity": dict(e_layers=3, d_model=512, d_ff=512, lr=5e-4, batch_size=16,
                            source="assumed"),
        "Traffic": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, batch_size=16,
                        source="assumed"),
        "Exchange": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
    },
}
for _name in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
    LEGACY["smamba"][_name], LEGACY["itransformer"][_name] = SMAMBA[_name], ITRANSFORMER[_name]

HORIZONS = {"PEMS03": [12, 24, 48, 96], "PEMS04": [12, 24, 48, 96],
            "PEMS07": [12, 24, 48, 96], "PEMS08": [12, 24, 48, 96]}
DEFAULT_HORIZONS = [96, 192, 336, 720]
TRAINING_KEYS = ("e_layers", "d_model", "d_ff", "d_state", "lr", "batch_size", "epochs")


def get_config(backbone, data, pred_len=96, version="official"):
    cfg = dict(COMMON, d_state=16)       # every S-Mamba entry sets d_state; unused by iTransformer
    if version == "legacy":
        cfg.update(LEGACY[backbone][data])
    elif version == "official":
        table, per_h = {"smamba": (SMAMBA, SMAMBA_H),
                        "itransformer": (ITRANSFORMER, ITRANSFORMER_H)}[backbone]
        cfg.update(table[data])
        cfg.update(per_h.get(data, {}).get(pred_len, {}))
    else:
        raise ValueError(version)
    cfg["config_version"] = version
    return cfg


def horizons(data):
    return HORIZONS.get(data, DEFAULT_HORIZONS)


def changed(backbone, data, pred_len):
    """Training settings that differ between the official and the legacy table."""
    o, g = get_config(backbone, data, pred_len), get_config(backbone, data, pred_len, "legacy")
    keys = TRAINING_KEYS if backbone == "smamba" else [k for k in TRAINING_KEYS if k != "d_state"]
    return {k: (g[k], o[k]) for k in keys if o[k] != g[k]}
