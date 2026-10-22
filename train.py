"""Train a dense GPT, convert it to a loss-free balanced MoE, and continue."""
import os
# Fragmentation-friendly allocator first: MoE expert counts change shape every
# step, which fragments the CUDA cache badly without expandable segments.
# Must be set before the first CUDA call; an explicit user setting wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import warnings
# torch.cuda imports pynvml internally and emits a deprecation FutureWarning;
# nvidia-ml-py is already the installed backend, so silence only that noise.
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)
import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.triton_ops import TRITON_AVAILABLE, gelu, weighted_scatter

try:
    import pynvml
except ImportError:
    pynvml = None


@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 704
    n_layers: int = 12
    n_heads: int = 11
    d_ff: int = 2816
    max_seq_len: int = 1024
    dropout: float = 0.0
    n_experts: int = 8
    top_k: int = 2
    moe_backend: str = "reference"
    bias_lr: float = 0.002


class JsonlWriter:
    """Small file-like adapter that keeps scalar and vector metrics readable."""
    def __init__(self, file): self.file = file
    def write(self, row):
        self.file.write(json.dumps(row) + "\n")
    def flush(self): self.file.flush()


class NVMLSampler:
    """Low-frequency GPU sampler; it never runs nvidia-smi in the hot path."""
    def __init__(self, device):
        self.enabled = device == "cuda" and pynvml is not None
        self.handle = None
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            except Exception:
                self.enabled, self.handle = False, None

    def read(self):
        if not self.enabled:
            return {"available": False}
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            raw_name = pynvml.nvmlDeviceGetName(self.handle)
            name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            return {
                "available": True,
                "name": name,
                "gpu_utilization_pct": util.gpu,
                "memory_used_mb": round(mem.used / 2**20, 2),
                "memory_total_mb": round(mem.total / 2**20, 2),
                "power_w": round(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000, 2),
                "temperature_c": pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU),
                "torch_memory_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
                "torch_memory_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2),
                "torch_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            }
        except Exception as exc:
            return {"available": False, "error": type(exc).__name__}

    def close(self):
        if self.enabled:
            try: pynvml.nvmlShutdown()
            except Exception: pass


class DenseMLP(nn.Module):
    def __init__(self, cfg, use_triton=False):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.use_triton = use_triton
    def forward(self, x):
        return self.drop(self.proj(gelu(self.fc(x), self.use_triton)))


class LossFreeMoE(nn.Module):
    """Top-k MoE with bias-only, auxiliary-loss-free load balancing.

    The balancing bias changes expert *selection*. It is not differentiated and
    is not part of the loss. Combining weights always come from unbiased logits.
    """
    def __init__(self, dense_mlp, cfg):
        super().__init__()
        self.backend = cfg.moe_backend
        use_triton = self.backend == "triton"
        self.experts = nn.ModuleList([DenseMLP(cfg, use_triton=use_triton) for _ in range(cfg.n_experts)])
        for expert in self.experts:
            expert.load_state_dict(dense_mlp.state_dict())
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.register_buffer("last_load", torch.zeros(cfg.n_experts))
        self.n_experts, self.top_k = cfg.n_experts, cfg.top_k
        self.bias_lr = cfg.bias_lr

    @torch.no_grad()
    def _update_bias(self, counts, n_tokens):
        load = counts.float() / max(1, n_tokens)
        self.last_load.copy_(load)
        # Mean-zero update keeps the bias numerically centered. Underloaded
        # experts get a positive selection bonus; overloaded experts lose it.
        self.expert_bias.add_(self.bias_lr * (load.mean() - load))
        self.expert_bias.sub_(self.expert_bias.mean())

    def forward(self, x):
        b, t, c = x.shape
        flat = x.reshape(-1, c)
        logits = self.router(flat)
        unbiased_prob = F.softmax(logits, dim=-1)
        selection = logits + self.expert_bias.to(logits.dtype)
        chosen = selection.topk(self.top_k, dim=-1).indices
        counts = torch.bincount(chosen.reshape(-1), minlength=self.n_experts)
        if self.training:
            self._update_bias(counts, flat.shape[0] * self.top_k)

        # The reference path is easiest to read. The batched/triton path sorts
        # assignments once and evaluates all experts as two grouped batched
        # GEMMs, removing the Python expert loop and many tiny kernel launches.
        if self.backend == "reference":
            out = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                token, _ = torch.where(chosen == expert_id)
                if token.numel():
                    out.index_add_(0, token, expert(flat[token]) * unbiased_prob[token, expert_id].unsqueeze(-1))
            denom = unbiased_prob.gather(1, chosen).sum(dim=1, keepdim=True).clamp_min(1e-9)
        else:
            tokens = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(self.top_k)
            expert_ids = chosen.reshape(-1)
            order = torch.argsort(expert_ids)
            sorted_tokens, sorted_experts = tokens[order], expert_ids[order]
            counts = torch.bincount(sorted_experts, minlength=self.n_experts)
            # Fixed capacity avoids a GPU->CPU synchronization on every batch.
            max_count = int(counts.max().item())
            starts = torch.cumsum(counts, dim=0) - counts
            rank = torch.arange(sorted_experts.numel(), device=flat.device) - torch.repeat_interleave(starts, counts)
            packed = torch.zeros((self.n_experts, max_count, c), device=flat.device, dtype=flat.dtype)
            packed[sorted_experts, rank] = flat[sorted_tokens]
            packed_tokens = torch.full((self.n_experts, max_count), -1, device=flat.device, dtype=torch.long)
            packed_tokens[sorted_experts, rank] = sorted_tokens
            w1 = torch.stack([e.fc.weight for e in self.experts])
            b1 = torch.stack([e.fc.bias for e in self.experts])
            w2 = torch.stack([e.proj.weight for e in self.experts])
            b2 = torch.stack([e.proj.bias for e in self.experts])
            hidden = gelu(torch.bmm(packed, w1.transpose(1, 2)) + b1[:, None, :], self.backend == "triton")
            packed_out = torch.bmm(hidden, w2.transpose(1, 2)) + b2[:, None, :]
            valid = torch.arange(max_count, device=flat.device)[None, :] < counts[:, None]
            grouped = packed_out[valid]
            if grouped.shape[0] != sorted_tokens.numel():
                raise RuntimeError("expert packing lost token assignments")
            packed_experts = torch.arange(self.n_experts, device=flat.device)[:, None].expand_as(packed_tokens)
            scatter_tokens = packed_tokens[valid]
            scatter_experts = packed_experts[valid]
            weights = unbiased_prob[scatter_tokens, scatter_experts]
            out = weighted_scatter(grouped, scatter_tokens, weights,
                                   flat.shape[0], use_triton=self.backend == "triton")
        if self.backend == "reference":
            out = out / denom
        return out.view(b, t, c)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.n_heads, self.head_dim = cfg.n_heads, cfg.d_model // cfg.n_heads
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if x.is_cuda:
            # SDPA selects FlashAttention on supported shapes, with safe
            # efficient/math fallback. This avoids hard-coding an external
            # FlashAttention build that only supports newer architectures.
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            except ImportError:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.drop(self.out(y.transpose(1, 2).contiguous().view(b, t, c)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.ln2, self.mlp = nn.LayerNorm(cfg.d_model), DenseMLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, moe=False):
        super().__init__(); self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f, self.lm_head = nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_dense_weights()
        self.lm_head.weight = self.wte.weight
        if moe:
            for block in self.blocks: block.mlp = LossFreeMoE(block.mlp, cfg)

    def _init_dense_weights(self):
        # GPT-2 style init so step-0 loss starts near ln(vocab) instead of
        # ~25. Default nn.Embedding N(0,1) makes initial logits ~sqrt(d_model)
        # and the model overconfidently wrong. Residual projections use the
        # 1/sqrt(2*n_layers) scale so depth does not blow up activations.
        std, res_std = 0.02, 0.02 / (2 * self.cfg.n_layers) ** 0.5
        nn.init.normal_(self.wte.weight, std=std)
        nn.init.normal_(self.wpe.weight, std=std)
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=std)
            if block.attn.qkv.bias is not None: nn.init.zeros_(block.attn.qkv.bias)
            nn.init.normal_(block.attn.out.weight, std=res_std)
            if block.attn.out.bias is not None: nn.init.zeros_(block.attn.out.bias)
            mlp = block.mlp
            experts = mlp.experts if isinstance(mlp, LossFreeMoE) else (mlp,)
            for expert in experts:
                nn.init.normal_(expert.fc.weight, std=std)
                nn.init.zeros_(expert.fc.bias)
                nn.init.normal_(expert.proj.weight, std=res_std)
                nn.init.zeros_(expert.proj.bias)
            nn.init.ones_(block.ln1.weight); nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight); nn.init.zeros_(block.ln2.bias)
        nn.init.ones_(self.ln_f.weight); nn.init.zeros_(self.ln_f.bias)
    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.drop_emb(self.wte(idx) + self.wpe(torch.arange(t, device=idx.device)))
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def to_moe(self):
        # Clone all trained dense weights first. Then replace each trained MLP
        # with experts initialized from that exact MLP (no random expert start).
        # Unwrap first so this also works when called on a torch.compile wrapper
        # (whose state_dict uses `_orig_mod.` prefixes).
        src = self._orig_mod if hasattr(self, "_orig_mod") else self
        out = GPT(src.cfg, moe=False).to(next(src.parameters()).device)
        out.load_state_dict(src.state_dict())
        for source, block in zip(src.blocks, out.blocks):
            block.mlp = LossFreeMoE(source.mlp, src.cfg)
        return out


def parameter_counts(cfg):
    d, f, v, pos, layers = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len, cfg.n_layers
    embeddings = v * d + pos * d
    attention_per_layer = 4 * d * d + 4 * d
    norms_per_layer = 4 * d
    mlp_per_layer = 2 * d * f + f + d
    final_norm = 2 * d
    dense = embeddings + layers * (attention_per_layer + norms_per_layer + mlp_per_layer) + final_norm
    moe = embeddings + layers * (attention_per_layer + norms_per_layer + cfg.n_experts * mlp_per_layer + d * cfg.n_experts) + final_norm
    return dense, moe


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def maybe_compile(model, args):
    """Compile with inductor, minus the MoE-hostile bits.

    Expert assignment counts change every step, so inductor would record one
    cudagraph per distinct shape and grow VRAM until OOM. Cudagraphs off
    costs a little launch overhead and saves the run.
    """
    try:
        import torch._inductor.config as _icfg
        _icfg.triton.cudagraphs = False
    except Exception:
        pass
    return torch.compile(model, mode=args.compile_mode)


def restore_rng_state(payload):
    """Restore RNG streams saved by save_checkpoint so resumed runs repeat."""
    try:
        if "torch_rng" in payload: torch.set_rng_state(payload["torch_rng"].cpu())
        if "numpy_rng" in payload: np.random.set_state(payload["numpy_rng"])
        if "python_rng" in payload: random.setstate(payload["python_rng"])
        if "cuda_rng" in payload and torch.cuda.is_available():
            states = payload["cuda_rng"]
            states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in states]
            torch.cuda.set_rng_state_all(states)
    except Exception as exc:
        print(f"warning: could not restore RNG state ({type(exc).__name__}); continuing", flush=True)


def batch(data, bs, seq, device):
    if len(data) < seq + 1:
        raise ValueError(f"token split needs at least seq_len + 1 tokens ({seq + 1}); got {len(data)}")
    starts = torch.randint(0, len(data) - seq, (bs,)).tolist()
    x = torch.stack([torch.from_numpy(np.asarray(data[i:i + seq], dtype=np.int64)) for i in starts]).to(device)
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + seq + 1], dtype=np.int64)) for i in starts]).to(device)
    return x, y


def tune_batch_size(model, data, args, name, writer):
    """Find the fastest fitting batch size without changing model weights."""
    if not args.auto_batch_size:
        return int(args.batch_size)
    if args.device != "cuda":
        selected = args.batch_size_min
        writer.write({"type": "batch_size_search", "stage": name, "selected": selected,
                      "reason": "CPU auto uses configured minimum"}); writer.flush()
        return selected

    if args.batch_size_candidates:
        candidates = sorted({int(x.strip()) for x in args.batch_size_candidates.split(",") if x.strip()})
        candidates = [x for x in candidates if args.batch_size_min <= x <= args.batch_size_max]
    else:
        candidates = [x for x in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
                      if args.batch_size_min <= x <= args.batch_size_max]
    if not candidates:
        raise ValueError("No batch-size candidates in the requested range")
    best_size, best_speed = None, -1.0
    was_training = model.training
    model.eval()  # freezes loss-free router bias updates during all probes
    print(f"[{name}] auto-tuning batch size over {len(candidates)} candidates "
          f"(4 passes each, median of last 3)...", flush=True)
    print(f"  {'batch':>6}  {'tok/s':>9}  {'peak VRAM':>10}  verdict", flush=True)
    for candidate in candidates:
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            # Drop the first pass as warm-up; median throughput is more stable.
            speeds = []
            for pass_index in range(4):
                x, y = batch(data, candidate, args.seq_len, args.device)
                start = time.perf_counter()
                with args.autocast_context(): _, loss = model(x, y)
                loss.backward()
                torch.cuda.synchronize()
                measured_speed = x.numel() / max(time.perf_counter() - start, 1e-9)
                if pass_index > 0: speeds.append(measured_speed)
                model.zero_grad(set_to_none=True)
            allocated = torch.cuda.max_memory_allocated() / 2**20
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**20
            # Reserve space for lazy AdamW first/second moments, initialized
            # only on the first optimizer update after tuning — plus inductor
            # workspace when this stage will train compiled (the probe itself
            # runs eager, so it never sees that overhead).
            optimizer_reserve = sum(p.numel() for p in model.parameters()) * 8 / 2**20
            will_compile = args.compile and (name == "moe" or args.stage == "dense")
            compile_reserve = total * 0.10 if will_compile else 0.0
            headroom_ok = (allocated + optimizer_reserve + compile_reserve
                           <= total * args.batch_size_safety)
            speed = float(np.median(speeds))
            writer.write({"type": "batch_size_probe", "stage": name, "batch_size": candidate,
                          "tokens_per_sec": speed, "peak_memory_mb": allocated,
                          "gpu_memory_mb": total, "headroom_ok": headroom_ok})
            writer.flush()
            verdict = "ok" if headroom_ok else "over budget, skipped"
            print(f"  {candidate:>6}  {speed:>9,.0f}  {allocated:>7,.0f} MiB  {verdict}", flush=True)
            if headroom_ok and speed > best_speed:
                best_size, best_speed = candidate, speed
            if not headroom_ok:
                continue
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            writer.write({"type": "batch_size_probe", "stage": name, "batch_size": candidate,
                          "oom": True}); writer.flush(); model.zero_grad(set_to_none=True)
            print(f"  {candidate:>6}  {'—':>9}  {'—':>10}  OOM, skipped", flush=True)
            torch.cuda.empty_cache(); break
    model.train(was_training)
    if best_size is None:
        raise RuntimeError("No batch size fits; lower --seq-len or --batch-size-max")
    writer.write({"type": "batch_size_selected", "stage": name, "batch_size": best_size,
                  "tokens_per_sec": best_speed}); writer.flush()
    print(f"[{name}] batch size -> {best_size} ({best_speed:,.0f} tok/s)", flush=True)
    return best_size


def resolve_cadence(value, steps, lo, hi):
    """Default cadences scale with run length (~10% of steps, clamped)."""
    if value is not None:
        return value
    return max(lo, min(hi, steps // 10))


def train_stage(model, data, steps, args, name, writer, resume_state=None, start_step=1):
    model.train()
    warmup_steps = resolve_cadence(args.warmup_steps, steps, 50, 500)
    eval_every = resolve_cadence(args.eval_every, steps, 50, 500)
    log_every = resolve_cadence(args.log_every, steps, 10, 100)
    print(f"[{name}] steps={steps} warmup={warmup_steps} eval_every={eval_every} "
          f"log_every={log_every} batch={args.batch_size} seq={args.seq_len}", flush=True)
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad: continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    def lr_factor(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, steps - warmup_steps)))
        min_ratio = args.lr_min / args.lr
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    seen_tokens = 0
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        seen_tokens = int(resume_state.get("seen_tokens", 0))
        # Fast-forward the scheduler to the resume point (it steps once per
        # optimizer update, so apply start_step - 1 steps).
        # Note: LambdaLR state was just loaded, so no extra stepping is needed
        # when the saved scheduler already reflects completed steps; the loop
        # below continues from start_step.
    gpu = NVMLSampler(args.device)
    started = time.perf_counter(); window_start = started; window_tokens = 0
    # If resuming, `elapsed` restarts from zero; `tokens` continues from the
    # saved count so cumulative throughput stays meaningful across restarts.
    for step in range(start_step, steps + 1):
        x, y = batch(data, args.batch_size, args.seq_len, args.device)
        with args.autocast_context():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); scheduler.step()
        loss_value = loss.item()
        seen_tokens += x.numel(); window_tokens += x.numel()
        elapsed = time.perf_counter() - started
        row = {"stage": name, "step": step, "loss": loss_value, "lr": opt.param_groups[0]["lr"],
               "grad_norm": grad_norm, "tokens": seen_tokens,
               "tokens_per_sec": seen_tokens / max(elapsed, 1e-9),
               "elapsed_sec": elapsed}
        if name == "moe":
            layer_loads = torch.stack([b.mlp.last_load for b in model.blocks])
            layer_biases = torch.stack([b.mlp.expert_bias for b in model.blocks])
            loads = layer_loads.mean(0)
            row["expert_loads"] = loads.tolist(); row["expert_biases"] = layer_biases.mean(0).tolist()
            row["layer_expert_loads"] = layer_loads.tolist()
        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            model.eval(); val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    vx, vy = batch(args.val_data, args.batch_size, args.seq_len, args.device)
                    with args.autocast_context(): _, val_loss = model(vx, vy)
                    val_losses.append(val_loss.item())
            model.train()
            row["val_loss"] = float(np.mean(val_losses)); row["val_perplexity"] = float(math.exp(min(20.0, row["val_loss"])))
        # GPU telemetry is intentionally sparse: NVML calls are outside the
        # training hot path and happen only on the configured interval.
        if args.gpu_log_every > 0 and (step == start_step or step % args.gpu_log_every == 0 or step == steps):
            row["gpu"] = gpu.read()
        if step == start_step or step % log_every == 0 or step == steps:
            now = time.perf_counter()
            row["tokens_per_sec_window"] = window_tokens / max(now - window_start, 1e-9)
            extra = f" grad_norm={grad_norm:.3f} tok/s={row['tokens_per_sec_window']:.0f}"
            if name == "moe":
                extra += f" loads={[round(v, 3) for v in loads.tolist()]}"
            print(f"{name} step={step:6d}/{steps} loss={loss_value:.4f}{extra}", flush=True)
            window_tokens = 0; window_start = now
        writer.write(row); writer.flush()
    gpu.close()
    return {"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
            "seen_tokens": seen_tokens, "completed_steps": steps}


def save_checkpoint(path, model, optimizer_state, scheduler_state, cfg, args,
                     stage, completed_steps, seen_tokens, batch_size):
    """Save model + optimizer + RNG state so --resume can continue a stage."""
    raw = unwrap_model(model)
    payload = {
        "model": raw.state_dict(),
        "config": asdict(cfg),
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "stage": stage,
        "completed_steps": completed_steps,
        "seen_tokens": seen_tokens,
        "batch_size": batch_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "triton_available": TRITON_AVAILABLE,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--stage", choices=["dense", "moe", "all"], default="all")
    p.add_argument("--init", type=Path); p.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--dense-out", type=Path, default=None, help="where to save the dense artifact with --stage all (default: <out-stem>.dense<suffix>)")
    p.add_argument("--resume", type=Path, default=None, help="resume a single stage (dense or moe) from a checkpoint saved by this script")
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--dense-steps", type=int, default=2000); p.add_argument("--moe-steps", type=int, default=2000)
    p.add_argument("--batch-size", default="auto", help="integer, or auto to benchmark fitting batch sizes")
    p.add_argument("--batch-size-min", type=int, default=1); p.add_argument("--batch-size-max", type=int, default=256)
    p.add_argument("--batch-size-candidates", default="", help="optional comma-separated batch-size candidates")
    p.add_argument("--batch-size-safety", type=float, default=0.80, help="maximum fraction of GPU VRAM including optimizer reserve")
    p.add_argument("--seq-len", type=int, default=512); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--lr-min", type=float, default=3e-5); p.add_argument("--warmup-steps", type=int, default=None, help="default: ~10%% of steps, clamped to [50, 500]"); p.add_argument("--weight-decay", type=float, default=0.1); p.add_argument("--log-every", type=int, default=None, help="default: ~10%% of steps, clamped to [10, 100]")
    p.add_argument("--eval-every", type=int, default=None, help="default: ~10%% of steps, clamped to [50, 500]"); p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--gpu-log-every", type=int, default=100, help="sample NVML GPU metrics every N steps; 0 disables")
    p.add_argument("--d-model", type=int, default=704); p.add_argument("--n-layers", type=int, default=12); p.add_argument("--n-heads", type=int, default=11); p.add_argument("--d-ff", type=int, default=2816)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n-experts", type=int, default=8); p.add_argument("--top-k", type=int, default=2); p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--bias-lr", type=float, default=0.002)
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--moe-backend", choices=["reference", "triton"], default="reference")
    p.add_argument("--compile", action="store_true", help="compile the model with TorchInductor")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default",
                   help="inductor mode; max-autotune exhaustively benchmarks GEMMs (verbose first step, only worth it for very long runs)")
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.d_model % args.n_heads: raise ValueError("--d-model must be divisible by --n-heads")
    if args.precision == "auto": args.precision = "bf16" if args.device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if args.precision == "bf16" and args.device == "cuda": args.amp_dtype = torch.bfloat16
    else: args.amp_dtype = None
    args.autocast_context = (lambda: torch.autocast(device_type="cuda", dtype=args.amp_dtype)) if args.amp_dtype else contextlib.nullcontext
    args.auto_batch_size = args.batch_size == "auto"
    if not args.auto_batch_size: args.batch_size = int(args.batch_size)
    if not 0 < args.batch_size_safety < 1: raise ValueError("--batch-size-safety must be between 0 and 1")
    if not 0.0 <= args.dropout < 1.0: raise ValueError("--dropout must be in [0, 1)")
    if args.resume is not None and args.stage == "all":
        raise ValueError("--resume supports single stages only; use --stage dense or --stage moe with --resume")
    cfg = Config(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff, max_seq_len=args.seq_len, dropout=args.dropout, n_experts=args.n_experts, top_k=args.top_k, moe_backend=args.moe_backend, bias_lr=args.bias_lr)
    data = np.memmap(args.data, mode="r", dtype=np.uint16)
    if len(data) < 2 * (args.seq_len + 1): raise ValueError("dataset needs enough tokens for train and validation windows")
    if int(np.max(data)) >= cfg.vocab_size: raise ValueError("data contains token IDs outside model vocabulary")
    split = min(max(args.seq_len + 1, int(len(data) * 0.99)), len(data) - (args.seq_len + 1))
    args.val_data = data[split:]
    train_data = data[:split]
    args.out.parent.mkdir(parents=True, exist_ok=True); jsonl_path = args.out.with_suffix(".loss.jsonl")
    with open(jsonl_path, "w", newline="") as f:
        # JSON Lines is intentional: expert arrays remain losslessly recorded,
        # while index.html can also parse the scalar columns directly.
        dense_params, moe_params = parameter_counts(cfg)
        writer = JsonlWriter(f); writer.write({"type": "run", "config": asdict(cfg), "device": args.device,
                                               "dense_params": dense_params, "moe_params": moe_params,
                                               "data_tokens": len(data), "train_tokens": len(train_data), "val_tokens": len(args.val_data), "aux_loss": 0.0,
                                               "precision": args.precision, "compile": args.compile,
                                               "triton_available": TRITON_AVAILABLE,
                                               "gpu_log_every": args.gpu_log_every})
        if args.stage in ("dense", "all"):
            dense_total = args.dense_steps if args.stage == "all" else args.steps
            dense_resume, dense_start = None, 1
            if args.stage == "dense" and args.resume is not None:
                payload = torch.load(args.resume, map_location=args.device, weights_only=False)
                if payload.get("stage") != "dense":
                    raise ValueError(f"--resume checkpoint stage is {payload.get('stage')!r}, expected 'dense'")
                saved_cfg = payload.get("config", {})
                shape_keys = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len")
                mismatched = [key for key in shape_keys if key in saved_cfg and saved_cfg[key] != getattr(cfg, key)]
                if mismatched:
                    raise ValueError(f"resume checkpoint config mismatch for {mismatched}")
                model = GPT(cfg).to(args.device)
                model.load_state_dict(payload["model"])
                dense_resume, dense_start = payload, int(payload.get("completed_steps", 0)) + 1
                restore_rng_state(payload)
                if isinstance(payload.get("batch_size"), int):
                    args.batch_size = int(payload["batch_size"]); args.auto_batch_size = False
                print(f"resuming dense from {args.resume} at step {dense_start}")
                print(f"device={args.device} dense_params={sum(p.numel() for p in model.parameters()):,}")
            else:
                if args.stage == "all" and args.resume is not None:
                    raise ValueError("--resume with --stage all is not supported; resume a single stage instead")
                model = GPT(cfg).to(args.device)
                print(f"device={args.device} dense_params={sum(p.numel() for p in model.parameters()):,}")
                args.batch_size = tune_batch_size(model, train_data, args, "dense", writer)
            if args.compile and args.stage == "dense": model = maybe_compile(model, args)
            if dense_resume is not None and dense_start > dense_total:
                print(f"dense checkpoint already complete ({dense_start - 1}/{dense_total}); nothing to do")
                state = {"optimizer": dense_resume["optimizer"], "scheduler": dense_resume["scheduler"],
                         "seen_tokens": dense_resume.get("seen_tokens", 0)}
            else:
                state = train_stage(model, train_data, dense_total, args, "dense", writer,
                                    resume_state=dense_resume, start_step=dense_start)
            if args.stage == "all":
                if args.dense_out is not None:
                    dense_checkpoint = args.dense_out
                else:
                    dense_checkpoint = args.out.with_name(args.out.stem + ".dense" + args.out.suffix)
            else:
                if args.dense_out is not None:
                    raise ValueError("--dense-out only applies to --stage all (use --out for single stages)")
                dense_checkpoint = args.out
            save_checkpoint(dense_checkpoint, model, state["optimizer"], state["scheduler"],
                            cfg, args, "dense", dense_total, state["seen_tokens"], args.batch_size)
            if args.stage == "all": print(f"saved dense checkpoint={dense_checkpoint}")
        if args.stage in ("moe", "all"):
            resume_state, resume_start = None, 1
            if args.stage == "moe":
                if args.resume is not None:
                    payload = torch.load(args.resume, map_location=args.device, weights_only=False)
                    if payload.get("stage") != "moe":
                        raise ValueError(f"--resume checkpoint stage is {payload.get('stage')!r}, expected 'moe'")
                    saved_cfg = payload.get("config", {})
                    shape_keys = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len")
                    mismatched = [key for key in shape_keys if key in saved_cfg and saved_cfg[key] != getattr(cfg, key)]
                    if mismatched:
                        raise ValueError(f"resume checkpoint config mismatch for {mismatched}")
                    model = GPT(cfg, moe=True).to(args.device)
                    model.load_state_dict(payload["model"])
                    resume_state, resume_start = payload, int(payload.get("completed_steps", 0)) + 1
                    restore_rng_state(payload)
                    args.batch_size = int(payload.get("batch_size", args.batch_size)) if not args.auto_batch_size or isinstance(payload.get("batch_size"), int) else args.batch_size
                    if args.auto_batch_size and isinstance(payload.get("batch_size"), int):
                        args.batch_size = int(payload["batch_size"]); args.auto_batch_size = False
                    print(f"resuming moe from {args.resume} at step {resume_start}")
                else:
                    if not args.init: raise ValueError("--init dense checkpoint is required for --stage moe")
                    payload = torch.load(args.init, map_location=args.device, weights_only=False)
                    saved_cfg = payload.get("config", {})
                    shape_keys = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len")
                    mismatched = [key for key in shape_keys if key in saved_cfg and saved_cfg[key] != getattr(cfg, key)]
                    if mismatched:
                        raise ValueError(f"dense checkpoint config mismatch for {mismatched}; use matching model dimensions and --seq-len")
                    dense = GPT(cfg).to(args.device); dense.load_state_dict(payload["model"])
                    model = dense.to_moe().to(args.device)
            else:
                if args.resume is not None:
                    raise ValueError("--resume with --stage all is not supported; resume a single stage instead")
                # Explicitly unwrap before conversion so a compiled dense model
                # cannot leak `_orig_mod.` prefixed weights into the MoE build.
                model = unwrap_model(model).to_moe().to(args.device)
            if resume_state is None:
                args.batch_size = tune_batch_size(model, train_data, args, "moe", writer)
            else:
                writer.write({"type": "batch_size_selected", "stage": "moe",
                              "batch_size": args.batch_size, "resumed": True}); writer.flush()
            if args.compile: model = maybe_compile(model, args)
            print(f"moe_params={sum(p.numel() for p in model.parameters()):,} (experts={cfg.n_experts}, top_k={cfg.top_k}, backend={cfg.moe_backend}, triton={TRITON_AVAILABLE}, aux_loss=0)")
            moe_total = args.moe_steps if args.stage == "all" else args.steps
            if resume_start > moe_total:
                print(f"moe checkpoint already complete ({resume_start - 1}/{moe_total}); nothing to do")
                state = {"optimizer": resume_state["optimizer"], "scheduler": resume_state["scheduler"],
                         "seen_tokens": resume_state.get("seen_tokens", 0)}
            else:
                state = train_stage(model, train_data, moe_total, args, "moe", writer,
                                    resume_state=resume_state, start_step=resume_start)
            save_checkpoint(args.out, model, state["optimizer"], state["scheduler"],
                            cfg, args, "moe", moe_total, state["seen_tokens"], args.batch_size)
    print(f"saved checkpoint={args.out} loss_log={jsonl_path}")


if __name__ == "__main__": main()
