"""Default hyperparameters per (backbone, dataset).

`source` records where each setting comes from:
- "S-Mamba script": copied from the official S-Mamba run scripts (checked).
- "assumed": a reasonable setting in the style of the official scripts that has NOT been
  checked against them; compare with the official repositories before reporting numbers.
Every value can be overridden from the command line.
"""

COMMON = dict(seq_len=96, dropout=0.1, activation="gelu", n_heads=8, use_norm=True,
              patience=3, batch_size=32, epochs=10)

SMAMBA = {
    "ETTh1": dict(e_layers=2, d_model=256, d_ff=256, d_state=2, lr=7e-5, source="S-Mamba script"),
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
    "PEMS03": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
    "PEMS04": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
    "PEMS07": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
    "PEMS08": dict(e_layers=4, d_model=512, d_ff=512, d_state=16, lr=1e-3, source="assumed"),
}

ITRANSFORMER = {
    "ETTh1": dict(e_layers=2, d_model=256, d_ff=256, lr=1e-4, source="assumed"),
    "ETTh2": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
    "ETTm1": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
    "ETTm2": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
    "Weather": dict(e_layers=3, d_model=512, d_ff=512, lr=1e-4, source="assumed"),
    "Electricity": dict(e_layers=3, d_model=512, d_ff=512, lr=5e-4, batch_size=16, source="assumed"),
    "Traffic": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, batch_size=16, source="assumed"),
    "Exchange": dict(e_layers=2, d_model=128, d_ff=128, lr=1e-4, source="assumed"),
    "PEMS03": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
    "PEMS04": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
    "PEMS07": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
    "PEMS08": dict(e_layers=4, d_model=512, d_ff=512, lr=1e-3, source="assumed"),
}

HORIZONS = {"PEMS03": [12, 24, 48, 96], "PEMS04": [12, 24, 48, 96],
            "PEMS07": [12, 24, 48, 96], "PEMS08": [12, 24, 48, 96]}
DEFAULT_HORIZONS = [96, 192, 336, 720]


def get_config(backbone, data):
    table = {"smamba": SMAMBA, "itransformer": ITRANSFORMER}[backbone]
    cfg = dict(COMMON, d_state=16)
    cfg.update(table[data])
    return cfg


def horizons(data):
    return HORIZONS.get(data, DEFAULT_HORIZONS)
