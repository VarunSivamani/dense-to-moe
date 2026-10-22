"""Generate text from a dense or MoE checkpoint (qualitative smoke check)."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from train import Config, GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", default="the quick brown fox")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--moe-backend", choices=["reference", "triton"], default="reference")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = payload.get("config", {})
    cfg = Config(**{**saved, "moe_backend": args.moe_backend}) if saved else Config()
    is_moe = any("experts" in k or "router" in k for k in payload["model"].keys())
    model = GPT(cfg, moe=is_moe).to(device).eval()
    model.load_state_dict(payload["model"])

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        idx = torch.tensor([tok.encode(args.prompt, add_special_tokens=False)],
                           dtype=torch.long, device=device)
        eos = tok.eos_token_id
        decode = tok.decode
    except Exception:
        idx = torch.tensor([[ord(c) % cfg.vocab_size for c in args.prompt]],
                           dtype=torch.long, device=device)
        eos, decode = None, lambda ids: "".join(chr(int(i) % 256) for i in ids)

    with torch.no_grad():
        for _ in range(args.tokens):
            logits, _ = model(idx[:, -cfg.max_seq_len:])
            logits = logits[:, -1, :] / max(1e-9, args.temperature)
            if args.top_k and args.top_k < logits.size(-1):
                cutoff = torch.topk(logits, args.top_k).values[:, -1:]
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos is not None and int(nxt.item()) == eos:
                break
    print(decode(idx[0].tolist()))


if __name__ == "__main__":
    main()
