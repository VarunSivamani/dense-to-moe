"""Stream a Hugging Face text dataset into a compact token memmap."""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--tokens", type=int, default=200_000_000)
    p.add_argument("--out", type=Path, default=Path("data/fineweb_200m.bin"))
    p.add_argument("--synthetic", action="store_true", help="offline smoke-test corpus")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for --synthetic")
    p.add_argument("--synthetic-vocab", type=int, default=2048, help="active vocabulary size for --synthetic")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        # Offline smoke-test corpus with real-text-like statistics: Zipf
        # unigram frequencies over a small active vocabulary plus ~35% local
        # copy (burstiness / n-gram repetition), so a transformer can actually
        # learn induction patterns. Seeded and reproducible.
        # Consequence: loss starts near ln(50257) ~= 10.82 (model is uniform
        # over the full vocab) and should fall toward the subset entropy
        # (~5-7 nats), NOT to zero. Compare smoke runs only against smoke runs.
        if not 1 <= args.synthetic_vocab <= np.iinfo(np.uint16).max:
            raise ValueError("--synthetic-vocab must fit in uint16")
        rng = np.random.default_rng(args.seed)
        active = args.synthetic_vocab
        ranks = np.arange(1, active + 1, dtype=np.float64)
        weights = 1.0 / ranks ** 1.15
        weights /= weights.sum()
        n = args.tokens
        ids = rng.choice(active, size=n, p=weights).astype(np.uint16)
        uniform = rng.random(n)
        backs = rng.integers(1, 129, size=n)
        for i in range(1, n):
            if uniform[i] < 0.35:
                b = int(backs[i]) if backs[i] <= i else i
                ids[i] = ids[i - b]
        ids.tofile(args.out)
        print(f"wrote {len(ids):,} synthetic tokens (active_vocab={active}, "
              f"unique={len(np.unique(ids))}, sample={ids[:16].tolist()}) to {args.out}")
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide an EOS token")
    kwargs = {"path": args.dataset, "split": args.split, "streaming": True}
    if args.config:
        kwargs["name"] = args.config
    ds = load_dataset(**kwargs)
    # GPT-2's vocabulary fits in uint16; fail loudly for an incompatible tokenizer.
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocabulary exceeds uint16; use uint32 storage")
    mm = np.memmap(args.out, mode="w+", dtype=np.uint16, shape=(args.tokens,))
    written = 0
    for row in tqdm(ds, total=None, desc="tokenizing"):
        ids = tokenizer.encode(row[args.text_column], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        take = min(len(ids), args.tokens - written)
        if take:
            mm[written:written + take] = ids[:take]
            written += take
        if written >= args.tokens:
            break
    mm.flush()
    # A partial final file is still valid and avoids silently training on zeros.
    if written < args.tokens:
        del mm
        with open(args.out, "r+b") as f:
            f.truncate(written * np.dtype(np.uint16).itemsize)
    print(f"wrote {written:,} tokens ({written * 2 / 1024**2:.1f} MiB) to {args.out}")


if __name__ == "__main__":
    main()
