"""CPU-only parity checks for dense -> MoE conversion and MoE backends.

Run with:  python tests/test_moe_parity.py
(No pytest dependency; exits non-zero on failure.)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import Config, GPT


def tiny_cfg(**over):
    kw = dict(d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32,
              n_experts=4, top_k=2, moe_backend="reference", bias_lr=0.002,
              dropout=0.0)
    kw.update(over)
    return Config(**kw)


def test_conversion_identity():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    dense = GPT(cfg).eval()
    moe = dense.to_moe().eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        dense_logits, _ = dense(idx)
        moe_logits, _ = moe(idx)
    max_diff = (dense_logits - moe_logits).abs().max().item()
    assert max_diff < 1e-5, f"dense->MoE identity broken: max_diff={max_diff}"
    print(f"PASS conversion_identity max_diff={max_diff:.2e}")


def test_reference_vs_batched():
    cfg_ref = tiny_cfg(moe_backend="reference")
    cfg_batched = tiny_cfg(moe_backend="triton")  # CPU fallback still uses batched GEMMs
    torch.manual_seed(1)
    ref = GPT(cfg_ref, moe=True).eval()
    torch.manual_seed(1)
    batched = GPT(cfg_batched, moe=True).eval()
    idx = torch.randint(0, cfg_ref.vocab_size, (2, 16))
    with torch.no_grad():
        logits_ref, _ = ref(idx)
        logits_batched, _ = batched(idx)
    max_diff = (logits_ref - logits_batched).abs().max().item()
    assert max_diff < 1e-4, f"reference vs batched mismatch: max_diff={max_diff}"
    print(f"PASS reference_vs_batched max_diff={max_diff:.2e}")


def test_bias_direction():
    torch.manual_seed(2)
    cfg = tiny_cfg()
    dense = GPT(cfg).eval()
    moe_model = GPT(cfg, moe=True)
    moe = moe_model.blocks[0].mlp
    moe.train()  # bias updates only in training mode
    n = cfg.n_experts
    counts = torch.zeros(n, dtype=torch.long)
    counts[0] = 100  # expert 0 heavily overloaded
    before = moe.expert_bias.clone()
    moe._update_bias(counts, int(counts.sum()))
    assert moe.expert_bias[0].item() < before[0].item(), "overloaded expert bias should decrease"
    assert abs(float(moe.expert_bias.mean().item())) < 1e-6, "bias should stay mean-zero"
    print(f"PASS bias_direction bias={moe.expert_bias.tolist()}")


def test_eval_freezes_bias():
    torch.manual_seed(3)
    cfg = tiny_cfg()
    dense = GPT(cfg).eval()
    moe = dense.to_moe()
    moe.eval()
    before = [b.mlp.expert_bias.clone() for b in moe.blocks]
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        moe(idx)
    for b, old in zip(moe.blocks, before):
        assert torch.equal(b.mlp.expert_bias, old), "eval forward must not mutate router bias"
    print("PASS eval_freezes_bias")


if __name__ == "__main__":
    test_conversion_identity()
    test_reference_vs_batched()
    test_bias_direction()
    test_eval_freezes_bias()
    print("All parity tests passed.")
