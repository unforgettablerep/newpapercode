"""Do learned identity embeddings reflect how the variables co-move?

python tools/analyze_identity.py checkpoints/smamba_Electricity_96_add_input_r8_p0.0_v1.0_s0.pt --root-path ./data

Compares, over all variable pairs, the cosine similarity of the learned identity vectors with
the Pearson correlation of the training-split series (Spearman rank correlation of the two),
and reports how often a variable's nearest identity neighbour is among its 10 most
correlated variables (vs. the chance rate).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idembed.data import load_splits  # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--root-path", default="./data")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]
    if "identity.0.table" in sd:
        E = sd["identity.0.table"]
    elif "identity.0.U" in sd:
        E = sd["identity.0.U"] @ sd["identity.0.V"]
    else:
        raise SystemExit("checkpoint has no input-level identity embedding")
    E = torch.nn.functional.normalize(E.float(), dim=1).numpy()
    (train, _, _), _ = load_splits(cfg["data"], a.root_path, cfg["seq_len"], cfg["pred_len"], False)
    X = train.values.numpy()
    corr = np.corrcoef(X.T)
    sim = E @ E.T
    n = len(E)
    iu = np.triu_indices(n, 1)
    print(f"variables: {n}, identity dim: {E.shape[1]}")
    print(f"Spearman(identity cosine, data correlation) over pairs: {spearman(sim[iu], corr[iu]):.3f}")
    np.fill_diagonal(sim, -np.inf)
    np.fill_diagonal(corr, -np.inf)
    k = min(10, n - 1)
    top = np.argsort(-corr, 1)[:, :k]
    nn_ = np.argmax(sim, 1)
    hit = np.mean([nn_[i] in top[i] for i in range(n)])
    print(f"nearest identity neighbour within the {k} most correlated variables: {hit:.2f} "
          f"(chance {k / (n - 1):.2f})")


if __name__ == "__main__":
    main()
