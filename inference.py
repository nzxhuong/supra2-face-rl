#!/usr/bin/env python3
"""
Supra2-IMG inference — standalone text-to-image (DiT + Flan-T5-Base + SD VAE).

Works on Linux and Windows. No imports from other project files.

Usage:
    python inference.py --prompt "a sea jellyfish floating in the pitch-black ocean depths" \\
        --seed 0 --cfg 3.0 --steps 50 --n 1 --out jellyfish.png

By default it uses your finetuned weights from ./runs/portrait (model_final_ema.pt if
training finished, otherwise the newest ckpt_step*.pt). Pick one explicitly with:
    python inference.py --ckpt runs/portrait/ckpt_step1000.pt --prompt "..."
Use --ckpt model_final_ema.pt for the original released model (a missing base file is
downloaded from Hugging Face: SupraLabs/Supra2-IMG).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Architecture constants (must match the trained checkpoint)
# ---------------------------------------------------------------------------
IMG_SIZE = 256
LATENT_SIZE = 32  # 256 / 8 (f8 VAE)
LATENT_CH = 4
PATCH = 2
NUM_TOKENS = (LATENT_SIZE // PATCH) ** 2

D_MODEL = 576
DEPTH = 14
N_HEADS = 9
MLP_RATIO = 4.0
D_CTX = 768  # Flan-T5-Base
MAX_CTX_LEN = 128
T5_NAME = "google/flan-t5-base"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
VAE_SCALE = 0.18215

HF_REPO = "SupraLabs/Supra2-IMG"
DEFAULT_CKPT = os.path.join(".", "model_final_ema.pt")  # original released weights
RUN_DIR = os.path.join(".", "runs", "portrait")  # --out folder used by supra2_finetune.py


def log(msg: str) -> None:
    """Print immediately so the user sees live progress."""
    print(msg, flush=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"[device] CUDA: {name} ({mem:.1f} GB)")
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log("[device] Apple MPS")
        return torch.device("mps")
    log("[device] CPU (this will be slow)")
    return torch.device("cpu")


# ===========================================================================
# MODEL
# ===========================================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class Attention(nn.Module):
    """Multi-head self- or cross-attention."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int | None = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_self = ctx_dim is None
        if self.is_self:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        if self.is_self:
            qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        else:
            M = ctx.shape[1]
            q = self.q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(ctx).view(B, M, 2, self.n_heads, self.head_dim)
            k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask.bool()[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    """DiT block: AdaLN-Zero self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, n_heads, ctx_dim=dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(6, dim=1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa)
        )
        x = x + self.cross_attn(self.norm_ca(x), ctx=ctx, ctx_mask=ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch, bias=True)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class SupraDiT(nn.Module):
    """~100M parameter DiT for rectified-flow text-to-image."""

    def __init__(
        self,
        latent_ch: int = LATENT_CH,
        d_model: int = D_MODEL,
        depth: int = DEPTH,
        n_heads: int = N_HEADS,
        ctx_dim: int = D_CTX,
        mlp_ratio: float = MLP_RATIO,
        num_tokens: int = NUM_TOKENS,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.patch = PATCH
        self.x_embed = nn.Linear(latent_ch * PATCH * PATCH, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.t_embed = TimestepEmbedder(d_model)
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, d_model, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(d_model, latent_ch * PATCH * PATCH)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = z.shape
        P = self.patch
        h, w = H // P, W // P
        x = z.view(B, C, h, P, w, P).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * P * P)
        x = self.x_embed(x) + self.pos_embed
        c = self.t_embed(t)
        ctx = self.ctx_proj(ctx)
        for blk in self.blocks:
            x = blk(x, c, ctx, ctx_mask)
        x = self.final(x, c)
        return x.view(B, h, w, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


# ===========================================================================
# Checkpoint download
# ===========================================================================
def ensure_checkpoint(path: str) -> str:
    """Use local checkpoint or download model_final_ema.pt from Hugging Face."""
    if os.path.isfile(path):
        log(f"[ckpt] found {path}")
        return path

    # Only the base model may be downloaded. A missing finetuned checkpoint must be an
    # error, otherwise the base weights would silently stand in for it.
    if os.path.abspath(path) != os.path.abspath(DEFAULT_CKPT):
        raise SystemExit(f"[ckpt] {path} not found (check the path / --ckpt)")

    log(f"[ckpt] {path} not found — downloading from {HF_REPO} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log("[ckpt] installing huggingface_hub ...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    filename = os.path.basename(path) or "model_final_ema.pt"
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    # Prefer the expected local path if the hub placed it elsewhere
    if os.path.isfile(filename) and os.path.abspath(filename) != os.path.abspath(path):
        import shutil

        shutil.copy2(filename, path)
        log(f"[ckpt] copied to {path}")
        return path
    log(f"[ckpt] downloaded: {downloaded}")
    return downloaded if os.path.isfile(downloaded) else path


def find_trained_ckpt() -> str:
    """Finished finetune if present, else the newest intermediate checkpoint, else the base model."""
    import glob
    import re

    final = os.path.join(RUN_DIR, "model_final_ema.pt")
    if os.path.isfile(final):
        return final
    steps = {}
    for f in glob.glob(os.path.join(RUN_DIR, "ckpt_step*.pt")):
        m = re.search(r"ckpt_step(\d+)\.pt$", f)
        if m:
            steps[int(m.group(1))] = f
    if steps:
        return steps[max(steps)]
    return DEFAULT_CKPT


# ===========================================================================
# Sampling (Euler integration of the flow ODE + optional CFG)
# ===========================================================================
@torch.no_grad()
def generate(args: argparse.Namespace, device: torch.device) -> None:
    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import AutoencoderKL
    import torchvision.utils as vutils

    ckpt_path = ensure_checkpoint(args.ckpt or find_trained_ckpt())

    log("[model] building SupraDiT ...")
    model = SupraDiT().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] {n_params / 1e6:.1f}M parameters")

    log(f"[model] loading weights from {ckpt_path} ...")
    t0 = time.perf_counter()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # cpu: skips optimizer state on GPU
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    if isinstance(state, dict):
        if cfg.get("patch", PATCH) != PATCH:
            raise SystemExit(f"Checkpoint PATCH={cfg['patch']} != script PATCH={PATCH}")
        weights = state["ema"] if "ema" in state else state.get("model", state)
    else:
        weights = state
    model.load_state_dict(weights, strict=True)
    if isinstance(state, dict) and "step" in state:
        log(f"[model] checkpoint was saved at training step {state['step']}")
    del weights, state
    log(f"[model] weights loaded in {time.perf_counter() - t0:.1f}s")

    ctx_len = int(cfg.get("ctx_len", MAX_CTX_LEN))
    log(f"[text] ctx_len={ctx_len}")

    log(f"[text] loading tokenizer + {T5_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
    text_model = T5EncoderModel.from_pretrained(T5_NAME).to(device).eval()
    for p in text_model.parameters():
        p.requires_grad = False

    log(f"[vae] loading {VAE_NAME} ...")
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    prompts = [args.prompt] * args.n
    n_tok = len(tokenizer(args.prompt)["input_ids"])
    log(f"[text] prompt tokens={n_tok}  n={args.n}  seed={args.seed}  cfg={args.cfg}  steps={args.steps}")
    if n_tok > ctx_len:
        log(f"[text] WARNING: prompt has {n_tok} tokens, truncated to ctx_len={ctx_len}")

    tok = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=ctx_len,
        return_tensors="pt",
    ).to(device)

    use_amp = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        ctx = text_model(**tok).last_hidden_state.float()
    cmask = tok["attention_mask"].float()

    use_cfg = args.cfg > 1.0
    if use_cfg:
        if isinstance(cfg, dict) and "uncond_text" in cfg:
            uncond_ctx = cfg["uncond_text"].to(device).float().unsqueeze(0).expand(args.n, -1, -1)
            uncond_mask = cfg["uncond_mask"].to(device).float().unsqueeze(0).expand(args.n, -1)
            log("[cfg] using stored unconditional embeddings")
        else:
            u_tok = tokenizer(
                [""] * args.n,
                padding="max_length",
                truncation=True,
                max_length=ctx_len,
                return_tensors="pt",
            ).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                uncond_ctx = text_model(**u_tok).last_hidden_state.float()
            uncond_mask = u_tok["attention_mask"].float()
            log("[cfg] encoded empty string as unconditional")
        ctx_all = torch.cat([ctx, uncond_ctx], 0)
        mask_all = torch.cat([cmask, uncond_mask], 0)

    z = torch.randn(args.n, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device=device)
    dt = 1.0 / args.steps
    log(f"[sample] Euler flow, {args.steps} steps ...")
    t_sample = time.perf_counter()

    for i in tqdm(range(args.steps), desc="Generating", unit="step", dynamic_ncols=True):
        t = torch.full((args.n,), i * dt, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if use_cfg:
                v_both = model(torch.cat([z, z], 0), torch.cat([t, t], 0), ctx_all, mask_all)
                v_cond, v_uncond = v_both.float().chunk(2, 0)
                v = v_uncond + args.cfg * (v_cond - v_uncond)
            else:
                v = model(z, t, ctx, cmask).float()
        z = z + dt * v

    log(f"[sample] denoising done in {time.perf_counter() - t_sample:.1f}s")
    log("[vae] decoding latents ...")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        imgs = vae.decode(z / VAE_SCALE).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(imgs, args.out, nrow=int(math.ceil(math.sqrt(args.n))))
    log(f"[done] saved {args.n} image(s) -> {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supra2-IMG standalone inference")
    p.add_argument(
        "--prompt",
        default="a sea jellyfish floating in the pitch-black ocean depths",
        help="Text prompt",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance scale")
    p.add_argument("--steps", type=int, default=50, help="Euler ODE steps")
    p.add_argument("--n", type=int, default=1, help="Number of images")
    p.add_argument("--out", default="jellyfish.png", help="Output image path")
    p.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint to load (default: newest finetuned one in ./runs/portrait, else the base model)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log("=== Supra2-IMG inference ===")
    device = pick_device()
    generate(args, device)


if __name__ == "__main__":
    main()