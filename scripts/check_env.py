"""Environment check for the GPU machine (e.g. an RTX 4090).

python scripts/check_env.py

1. prints PyTorch / CUDA / GPU information and whether mamba_ssm is importable;
2. loads the weights of an official mamba_ssm.Mamba block into the pure-PyTorch MambaTorch
   (identical parameter names) and compares their outputs — this validates both paths;
3. runs one training step of S-Mamba with identity embeddings at Electricity size and
   reports the time and peak memory.
"""
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idembed.mamba import MambaTorch, cuda_mamba_available  # noqa: E402
from idembed.models import VariateForecaster  # noqa: E402


def main():
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("no GPU found; the code will run on CPU with the pure-PyTorch Mamba (slow)")
        return
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    print(f"mamba_ssm importable: {cuda_mamba_available()}")

    if cuda_mamba_available():
        from mamba_ssm import Mamba
        torch.manual_seed(0)
        ref = Mamba(d_model=512, d_state=16, d_conv=2, expand=1).to(dev)
        alt = MambaTorch(512, d_state=16, d_conv=2, expand=1).to(dev)
        missing = alt.load_state_dict(ref.state_dict(), strict=False)
        print(f"weight transfer: missing {missing.missing_keys}, unexpected {missing.unexpected_keys}")
        x = torch.randn(4, 325, 512, device=dev)
        with torch.no_grad():
            a, b = ref(x), alt(x)
        print(f"mamba_ssm vs MambaTorch: max |diff| {(a - b).abs().max().item():.2e} "
              f"(output scale {a.abs().mean().item():.2e})")
    else:
        print("install mamba_ssm for the fast path: pip install causal-conv1d mamba-ssm "
              "(see README); falling back to MambaTorch")

    model = VariateForecaster("smamba", 321, 96, 96, d_model=512, d_ff=512, e_layers=3,
                              d_state=16, identity="add").to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    x, m, y = torch.randn(16, 96, 321, device=dev), torch.randn(16, 96, 4, device=dev), torch.randn(16, 96, 321, device=dev)
    for i in range(6):
        if i == 1:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
        loss = torch.nn.functional.mse_loss(model(x, m), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"S-Mamba+identity, Electricity size (B=16, 321+4 tokens, d=512, 3 layers): "
          f"{(time.time() - t0) / 5 * 1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
