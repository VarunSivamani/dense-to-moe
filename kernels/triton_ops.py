"""Small, autograd-safe Triton primitives for the MoE hot path.

The kernel is optional: importing this module never makes CPU/reference runs
depend on Triton. Forward and backward both use Triton when available; the
Python fallback is mathematically identical up to normal floating-point error.
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    # Fixed tile sizes on purpose: @triton.autotune re-benchmarks on every new
    # shape, and MoE shapes change every step (imbalanced expert counts), so
    # autotuning costs far more than it saves at this scale. Inductor still
    # tunes the big GEMMs when --compile is on.
    @triton.jit
    def _gelu_forward(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))
        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.jit
    def _gelu_backward(x_ptr, dy_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
        cdf = 0.5 * (1.0 + tl.erf(x / 1.4142135623730951))
        pdf = 0.3989422804014327 * tl.exp(-0.5 * x * x)
        tl.store(dx_ptr + offsets, dy * (cdf + x * pdf), mask=mask)

    @triton.jit
    def _weighted_scatter(grouped_ptr, token_ptr, weight_ptr, out_ptr, denom_ptr,
                          total, hidden, BLOCK: tl.constexpr):
        """Scatter grouped expert outputs and router weights in one kernel."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total * hidden
        assignment = offsets // hidden
        column = offsets % hidden
        token = tl.load(token_ptr + assignment, mask=mask, other=0).to(tl.int32)
        weight = tl.load(weight_ptr + assignment, mask=mask, other=0.0)
        value = (tl.load(grouped_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                 * weight.to(tl.float32))
        tl.atomic_add(out_ptr + token * hidden + column, value, mask=mask)
        # One lane per assignment is enough to accumulate the denominator.
        lane = (offsets % hidden) == 0
        tl.atomic_add(denom_ptr + token, weight, mask=mask & lane)


class _TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.empty_like(x)
        _gelu_forward[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        _gelu_backward[(triton.cdiv(x.numel(), 1024),)](x, dy, dx, x.numel(), BLOCK=1024)
        return dx


def gelu(x, use_triton=False):
    if use_triton and TRITON_AVAILABLE and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return _TritonGELU.apply(x)
    return F.gelu(x)


class _TritonWeightedScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grouped, token_ids, weights, n_tokens):
        total, hidden = grouped.shape
        denom = torch.zeros((n_tokens,), device=grouped.device, dtype=torch.float32)
        out = torch.zeros((n_tokens, hidden), device=grouped.device, dtype=torch.float32)
        _weighted_scatter[(triton.cdiv(total * hidden, 256),)](grouped, token_ids, weights, out, denom,
                                                            total, hidden, BLOCK=256)
        denom = denom.clamp_min(1e-9)
        normalized = (out / denom[:, None]).to(grouped.dtype)
        ctx.save_for_backward(grouped, token_ids, weights, normalized, denom)
        return normalized

    @staticmethod
    def backward(ctx, grad_out):
        grouped, token_ids, weights, out, denom = ctx.saved_tensors
        token_grad = grad_out[token_ids]
        token_out = out[token_ids]
        scale = denom[token_ids, None]
        grad_grouped = (token_grad * (weights[:, None] / scale)).to(grouped.dtype)
        grad_weights = (token_grad.float() * (grouped.float() - token_out.float())).sum(-1) / denom[token_ids]
        grad_weights = grad_weights.to(weights.dtype)
        return grad_grouped, None, grad_weights, None


def weighted_scatter(grouped, token_ids, weights, n_tokens, use_triton=False):
    """Combine grouped expert results, retaining gradients for training."""
    if use_triton and TRITON_AVAILABLE and grouped.is_cuda:
        return _TritonWeightedScatter.apply(grouped, token_ids, weights, n_tokens)
    out = torch.zeros((n_tokens, grouped.shape[-1]), device=grouped.device, dtype=grouped.dtype)
    denom = torch.zeros((n_tokens,), device=grouped.device, dtype=grouped.dtype)
    out.index_add_(0, token_ids, grouped * weights[:, None])
    denom.index_add_(0, token_ids, weights)
    return out / denom.clamp_min(1e-9)[:, None]
