# Dense GPT → auxiliary-loss-free MoE

Decoder-only GPT pretraining, exact-copy dense→MoE upcycling, and continued
training with DeepSeek-style bias-only (aux-loss-free) load balancing. Single
process, one GPU. The training objective is always plain next-token
cross-entropy — routing uses `router_logits + expert_bias` for *selection*
but unbiased softmax probabilities for *weighting*, so `aux_loss` is 0.

Pipeline: `prepare_data.py` → `train.py` → `sample.py` / `index.html`.

## Top command — dense + MoE in one run

```bash
python train.py --data data/fineweb_200m.bin --stage all \
  --out checkpoints/moe.pt --dense-out checkpoints/dense.pt \
  --dense-steps 2000 --moe-steps 2000 \
  --precision bf16 --moe-backend triton --compile --batch-size auto
```

Trains dense, converts every MLP to 8 exact-copy experts (no loss spike),
and continues as MoE. Final artifacts: `checkpoints/dense.pt`,
`checkpoints/moe.pt`, metrics at `checkpoints/moe.loss.jsonl`.
Without `--dense-out`, the dense artifact defaults to
`<out-stem>.dense<suffix>`. Smoke version (no download, ~1 min):

```bash
python prepare_data.py --synthetic --out data/smoke.bin --tokens 200000
python train.py --data data/smoke.bin --stage all \
  --out checkpoints/moe.pt --dense-out checkpoints/dense.pt \
  --dense-steps 30 --moe-steps 30 --batch-size 4 --seq-len 128 --eval-every 10
```

Short on time? The fast variant skips the compile autotune tax and uses the
reference MoE path (often quicker at small batches):

```bash
python train.py --data data/fineweb_200m.bin --stage all \
  --out checkpoints/moe.pt --dense-out checkpoints/dense.pt \
  --dense-steps 2000 --moe-steps 2000 \
  --precision bf16 --moe-backend reference --batch-size 32
```

## Install

```bash
pip install -r requirements.txt
```

## The continuous workflow

Run these in order. `<data>` is your token file, `<out>` your checkpoint
prefix. Every step appends to the same JSONL log next to the checkpoint.

**Step 0 — make data** (once per corpus):

```bash
# Real corpus: stream FineWeb-Edu, tokenize with GPT-2 BPE, write 200M uint16 tokens (~381.5 MiB)
python prepare_data.py --out data/fineweb_200m.bin --tokens 200000000

# Offline smoke corpus (no download): seeded Zipf + trigram-copy over a 2048-token active vocab
python prepare_data.py --synthetic --out data/smoke.bin --tokens 200000
```

**Step 1 — train dense:**

```bash
python train.py --data data/fineweb_200m.bin --stage dense \
  --out checkpoints/dense.pt --steps 2000 \
  --precision bf16 --batch-size auto
```

**Step 2 — convert + continue as MoE** (each expert starts as an exact copy
of the dense MLP, so there is no loss spike):

```bash
python train.py --data data/fineweb_200m.bin --stage moe \
  --init checkpoints/dense.pt --out checkpoints/moe.pt --steps 2000 \
  --precision bf16 --moe-backend triton --compile --batch-size auto
```

Or steps 1–2 in a single invocation (final artifacts `checkpoints/dense.pt` +
`checkpoints/moe.pt` via `--dense-out`):

```bash
python train.py --data data/fineweb_200m.bin --stage all \
  --out checkpoints/moe.pt --dense-out checkpoints/dense.pt \
  --dense-steps 2000 --moe-steps 2000 \
  --precision bf16 --moe-backend triton --compile --batch-size auto
```

**Step 3 — keep going / recover** (single stages only; continues step
numbering, LR schedule, token counts, and RNG):

```bash
python train.py --data data/fineweb_200m.bin --stage moe \
  --resume checkpoints/moe.pt --out checkpoints/moe.pt --steps 40000 \
  --batch-size 32 --seq-len 512
```

**Step 4 — check the model:**

```bash
python tests/test_moe_parity.py
python sample.py --checkpoint checkpoints/moe.pt --prompt "the quick brown fox" --tokens 80
```

**Step 5 — visualize:** open `index.html`, drop in `<out>.loss.jsonl`
(file picker, drag-and-drop, or paste). See [Dashboard](#dashboard).

## Recipes

```bash
# Smoke test, CPU or small GPU (~1 min): loss 10.82 -> ~10.2, spike-free fork, MoE keeps falling
python prepare_data.py --synthetic --out data/smoke.bin --tokens 200000
python train.py --data data/smoke.bin --stage all --out checkpoints/demo.pt \
  --dense-steps 30 --moe-steps 30 --batch-size 4 --seq-len 128 --eval-every 10

# Small-GPU full run (fits ~24GB): narrower model, shorter context, manual batch
python train.py --data data/fineweb_200m.bin --stage all --out checkpoints/small.pt \
  --d-model 512 --n-layers 8 --n-heads 8 --d-ff 2048 \
  --seq-len 256 --batch-size 16 --dense-steps 2000 --moe-steps 2000

# A100 max-throughput: autotuned batch, Triton MoE, TorchInductor, sparse NVML
python train.py --data data/fineweb_200m.bin --stage all --out checkpoints/a100.pt \
  --dense-steps 2000 --moe-steps 2000 --precision bf16 \
  --moe-backend triton --compile --batch-size auto --gpu-log-every 100

# Backend ablation (same batch/seq/seed; compare tokens_per_sec_window + peak VRAM + loss after warmup)
python train.py --data data/fineweb_200m.bin --stage moe --init checkpoints/dense.pt \
  --out checkpoints/moe-ref.pt --steps 2000 --batch-size 32 --seq-len 512 --moe-backend reference
python train.py --data data/fineweb_200m.bin --stage moe --init checkpoints/dense.pt \
  --out checkpoints/moe-tri.pt --steps 2000 --batch-size 32 --seq-len 512 --moe-backend triton

# Custom MoE shape + regularization
python train.py --data data/fineweb_200m.bin --stage all --out checkpoints/custom.pt \
  --n-experts 4 --top-k 1 --bias-lr 0.005 --dropout 0.1 --weight-decay 0.05
```

Rule of thumb: `reference` without `--compile` for short runs and small
batches (no autotune tax, exact same math); `triton` + `--compile` for
long runs at large batches where the batched GEMMs amortize.

## All options — `train.py`

| Flag | Default | What it does |
|---|---|---|
| `--data` | (required) | Token `.bin` from `prepare_data.py` |
| `--stage` | `all` | `dense`: train dense only · `moe`: convert `--init` + train MoE · `all`: dense then MoE |
| `--init` | — | Dense checkpoint for `--stage moe` |
| `--out` | `checkpoints/model.pt` | Output checkpoint = the MoE artifact. Name it `checkpoints/moe.pt` |
| `--dense-out` | `<out-stem>.dense<suffix>` | Dense artifact path for `--stage all`. Pass `checkpoints/dense.pt` for canonical `dense.pt` + `moe.pt` finals |
| `--resume` | — | Continue `--stage dense`/`moe` from its own checkpoint (not with `all`) |
| `--steps` | `2000` | Steps for single-stage runs |
| `--dense-steps` / `--moe-steps` | `2000` / `2000` | Steps per stage for `--stage all` |
| `--batch-size` | `auto` | Integer, or `auto` to benchmark candidates |
| `--batch-size-min` / `--max` | `1` / `256` | Search bounds for `auto` |
| `--batch-size-candidates` | — | e.g. `"16,32,64"` to override the default grid |
| `--batch-size-safety` | `0.80` | Max fraction of VRAM (activations + AdamW reserve) |
| `--seq-len` | `512` | Context length. Part of checkpoint shape — reload with the same value |
| `--lr` / `--lr-min` | `3e-4` / `3e-5` | Cosine schedule endpoints |
| `--warmup-steps` | auto (`steps//10` in `[50,500]`) | Linear warmup before cosine decay |
| `--weight-decay` | `0.1` | AdamW decay on 2D weights only (biases/norms excluded) |
| `--log-every` | auto (`steps//10` in `[10,100]`) | Console + windowed-tok/s cadence |
| `--eval-every` / `--eval-batches` | auto (`steps//10` in `[50,500]`) / `10` | Val cadence on the held-out final 1% (`0` disables). Eval freezes router bias |
| `--gpu-log-every` | `100` | NVML sample cadence (`0` disables; never in the hot path) |
| `--d-model` / `--n-layers` / `--n-heads` / `--d-ff` | `704` / `12` / `11` / `2816` | Model shape. Keep `d_model % n_heads == 0`. Default ≈108M dense / ≈441M MoE (tied embeddings) |
| `--dropout` | `0.0` | Dropout on embeddings, attention out, MLP |
| `--n-experts` / `--top-k` | `8` / `2` | MoE shape |
| `--bias-lr` | `0.002` | Router-bias update rate (mean-zeroed, gradient-free) |
| `--seed` | `1337` | Python + NumPy + torch + CUDA seed |
| `--precision` | `auto` | `auto` → bf16 on capable CUDA, else fp32 |
| `--moe-backend` | `reference` | `reference` (readable loop) or `triton` (sorted batched GEMMs + fused kernels) |
| `--compile` | off | TorchInductor compile |
| `--compile-mode` | `default` | `default` (quiet, recommended) · `reduce-overhead` (cudagraphs) · `max-autotune` (exhaustive GEMM search: minutes of verbose autotune on the first MoE step, only pays off on 10k+ step runs) |

## All options — `prepare_data.py`

| Flag | Default | What it does |
|---|---|---|
| `--dataset` / `--config` / `--split` / `--text-column` | `HuggingFaceFW/fineweb-edu` / — / `train` / `text` | Streaming source |
| `--tokenizer` | `gpt2` | HF tokenizer; vocab must fit `uint16` |
| `--tokens` | `200000000` | Target token count (short corpora truncate cleanly) |
| `--out` | `data/fineweb_200m.bin` | Output memmap file |
| `--synthetic` | off | Offline smoke corpus (no download) |
| `--seed` | `1337` | Seed for `--synthetic` |
| `--synthetic-vocab` | `2048` | Active vocab for `--synthetic`. Loss starts at `ln(50257)≈10.82`, falls toward subset entropy (~5 nats) — compare smoke vs smoke only |

## All options — `sample.py`

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint` | (required) | Dense or MoE checkpoint (auto-detected) |
| `--prompt` | `the quick brown fox` | Starting text |
| `--tokens` | `80` | Tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling cutoff |
| `--tokenizer` | `gpt2` | Must match training tokenizer |
| `--moe-backend` | `reference` | Backend for MoE forward |
| `--seed` | `0` | Sampling seed |

## Files you get

- **Checkpoints** — canonical finals: `checkpoints/dense.pt` (dense),
  `checkpoints/moe.pt` (MoE). With `--stage all`, pass
  `--out checkpoints/moe.pt --dense-out checkpoints/dense.pt`; without
  `--dense-out` the dense artifact defaults to `<out-stem>.dense<suffix>`:
  model weights, config, optimizer + scheduler state, RNG streams,
  completed steps, batch size, torch version. Reload requirements: same
  `--d-model/--n-layers/--n-heads/--d-ff/--seq-len`; `--resume` additionally
  needs the same `--stage`.
- **Metrics** (`<out>.loss.jsonl`, one JSON object per line):
  - `{"type":"run", ...}` — config, device, `dense_params`/`moe_params`,
    token counts, `aux_loss: 0.0`, precision, compile/Triton flags.
  - `{"stage","step","loss","lr","grad_norm","tokens","tokens_per_sec",...}` —
    per training step (`tokens_per_sec` cumulative, `tokens_per_sec_window`
    interval); MoE rows add `expert_loads`, `expert_biases`,
    `layer_expert_loads`; eval steps add `val_loss`/`val_perplexity`;
    sparse steps add `gpu` (util, VRAM used/total, power, temp, torch mem).
  - `{"type":"batch_size_probe"/"batch_size_selected", ...}` — autotune
    record per candidate plus the winner.

## Dashboard

`index.html` is one self-contained file (no build, no network, no
dependencies). Three tabs:

1. **Training** — loss (dense + MoE + val, `ln(V)` baseline), grad + LR,
   throughput, GPU/VRAM, run manifest. Hover for values, click legend
   chips to isolate series, smoothing slider + log-scale toggle.
2. **Experts** — per-expert loads, layer×expert heatmap (scrub + play
   through MoE history), router bias, batch-probe table.
3. **Architecture** — playable miniature of your run config, no log needed.
   Click stack layers, toggle dense/MoE, fork one MLP into N experts, drag
   token chips (or type) into the sequence, Play to stream tokens through
   the stack with the same `top-k(router + bias)` rule as training
   (live mean-zero bias updates, disableable experts, event feed).
   Biases seed from your log's latest MoE row when present.

### Deploy (Netlify Drop) + embedding your run

The page loads data by priority: **embedded JSON > `./run.loss.jsonl`
sidecar > empty state** (drop a file, paste, or run the demo). To publish
a self-contained report: thin the log, inject it into the embed block,
and drag the single `index.html` onto Netlify Drop (a folder Drop with
`run.loss.jsonl` beside it works too, via the sidecar path):

```bash
python -c "
import json
rows=[json.loads(l) for l in open('checkpoints/moe.loss.jsonl')]
thin=[rows[0]]+[r for r in rows[1:] if r.get('type','').startswith('batch') or r.get('step',1)%2==0]
html=open('index.html').read().replace(
  '<script id=\"run-data\" type=\"application/json\">[]</script>',
  '<script id=\"run-data\" type=\"application/json\">'+json.dumps(thin)+'</script>', 1)
open('index.html','w').write(html)"
```

## Method notes (short)

- **Init**: GPT-2-style (`N(0,0.02)`, residual projs `÷√(2·n_layers)`), head
  tied to embeddings → step-1 loss starts at `ln(vocab)`.
- **Schedule**: linear warmup → cosine decay to `--lr-min`; gradient clip 1.0.
- **Batch autotune**: 4 fwd/bwd passes per candidate (first is warmup, median
  of rest wins); OOM and over-`--batch-size-safety` rejected; dense and MoE
  tuned independently with router bias frozen.
- **Eval**: final 1% of the token file, no grads, no bias updates.
- **Attention**: SDPA with Flash→efficient→math fallback under CUDA.
- **MoE cost**: top-2 routing does ~2× the MLP FLOPs of dense plus sort/pad
  overhead, so MoE tok/s is an order of magnitude below dense — expected.
  `--compile` defaults to quiet `default` mode; `max-autotune` prints pages
  of GEMM search on the first MoE step and only pays off on very long runs.
  Compiled MoE runs with cudagraphs off (expert shapes change every step;
  one graph per shape would OOM a long run).
- **Limits**: single process, one GPU; no DDP/multi-GPU use.

## Troubleshooting

- **OOM in tuning / first step** → lower `--batch-size-max` / `--seq-len` /
  `--batch-size-safety`, or pass an explicit small `--batch-size`.
- **OOM mid-run (e.g. step 40, not step 1)** → memory *growth*, not sizing:
  each new expert-count shape used to record another cudagraph. Current code
  disables cudagraphs under `--compile` and uses an
  fragmentation-friendly allocator by default; the tuner also reserves an
  extra ~10% VRAM for compiled stages. If you still grow, drop to a smaller
  manual `--batch-size`.
- **`config mismatch`** → checkpoint was built with different
  `--d-model/--n-layers/--n-heads/--d-ff/--seq-len`; rerun matching.
- **Triton silently falls back** → check `triton_available` in the run
  header and the `triton=...` MoE startup line; needs CUDA + `triton`.
- **Loss starts ~25, not ~10.8** → you are on the pre-init code; pull
  latest (init fix) — step 1 must equal `ln(vocab)`.
- **Resume says nothing to do** → checkpoint already reached `--steps`;
  raise `--steps` to continue further.
- **`--resume` + `--stage all`** → not supported; resume one stage at a time.
- **Smoke loss won't fall** → use `--synthetic` data for smoke (learnable)
  and compare smoke-vs-smoke only; real loss needs the HF corpus.
- **Inductor log flood on first MoE step** → normal for
  `--compile-mode max-autotune`; use default mode or drop `--compile` for
  short runs. `No valid triton configs ... Ignoring` lines are benign
  (falls back to cuBLAS, which usually wins anyway).
- **MoE much slower than dense** → expected (~2× FLOPs + routing overhead);
  try `--moe-backend reference` for small batches, and note the tuner's
  MoE tok/s is pre-compile (eager) so it understates compiled throughput.
