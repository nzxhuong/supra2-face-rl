#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from inference import (  # official file from the model repo
        IMG_SIZE, LATENT_CH, LATENT_SIZE, MAX_CTX_LEN, T5_NAME, VAE_NAME, VAE_SCALE,
        SupraDiT, ensure_checkpoint, pick_device,
    )
except ImportError:
    sys.exit("inference.py not found next to this script. Get it with:\n"
             "  wget https://huggingface.co/SupraLabs/Supra2-IMG/resolve/main/inference.py")

from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
D_CTX = 768


def log(msg: str) -> None:
    """Print without breaking an active progress bar."""
    tqdm.write(msg)


# --------------------------------------------------------------------------- helpers
def amp(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_t5(device):
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(T5_NAME)
    t5 = T5EncoderModel.from_pretrained(T5_NAME).to(device).eval()
    t5.requires_grad_(False)
    return tok, t5


@torch.no_grad()
def encode_text(tokenizer, t5, prompts, ctx_len, device):
    tok = tokenizer(prompts, padding="max_length", truncation=True,
                    max_length=ctx_len, return_tensors="pt").to(device)
    with amp(device):
        emb = t5(**tok).last_hidden_state
    return emb.float(), tok["attention_mask"]


def load_square(path, size, vcrop):
    """Resize short side to `size`, crop a square. vcrop: 0=top, 0.5=center (tall images)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = im.size
    left = (w - size) // 2
    top = int(round((h - size) * vcrop))
    im = im.crop((left, top, left + size, top + size))
    arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# --------------------------------------------------------------------------- prepare
def cmd_prepare(a):
    from diffusers import AutoencoderKL

    device = pick_device()
    src, out = Path(a.data), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    items, skipped = [], {"no_caption": 0, "too_small": 0, "unreadable": 0}
    best = {}  # caption path -> (quality key, image path, caption text)
    dup_lines = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        cap = p.with_suffix(".txt")
        text = cap.read_text(encoding="utf-8").strip() if cap.exists() else ""
        if not text:
            skipped["no_caption"] += 1
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            skipped["unreadable"] += 1
            continue
        if min(w, h) < a.min_size:
            skipped["too_small"] += 1
            continue
        # Several images can share one caption file (a.jpg + a.png -> a.txt). Keep the
        # best one per caption: more pixels first, then lossless (png) over lossy.
        key = (w * h, p.suffix.lower() == ".png")
        if cap in best:
            old_key, old_p, _ = best[cap]
            keep, drop = (p, old_p) if key > old_key else (old_p, p)
            dup_lines.append(f"kept {keep}\tdropped {drop}")
            if key > old_key:
                best[cap] = (key, p, text)
        else:
            best[cap] = (key, p, text)
    items = [(p, text) for _, p, text in best.values()]
    if dup_lines:
        (out / "duplicates.txt").write_text("\n".join(dup_lines) + "\n", encoding="utf-8")
        log(f"[prepare] {len(dup_lines)} images shared a caption file with another image; kept one "
            f"per caption (list in {out / 'duplicates.txt'}). If those were different photos that "
            f"just share a file name, rename them and re-caption instead.")
    if not items:
        sys.exit("no usable image+caption pairs found")
    N = len(items)
    log(f"[prepare] {N} usable pairs, skipped: {skipped}")

    tokenizer, t5 = load_t5(device)
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    vae.requires_grad_(False)

    mm = np.lib.format.open_memmap
    f16 = np.float16
    lat_shape = (N, LATENT_CH, LATENT_SIZE, LATENT_SIZE)
    mean_mm = mm(out / "latent_mean.npy", mode="w+", dtype=f16, shape=lat_shape)
    std_mm = mm(out / "latent_std.npy", mode="w+", dtype=f16, shape=lat_shape)
    emb_mm = mm(out / "text_emb.npy", mode="w+", dtype=f16, shape=(N, a.ctx_len, D_CTX))
    mask_mm = mm(out / "text_mask.npy", mode="w+", dtype=np.uint8, shape=(N, a.ctx_len))

    n_trunc = 0
    pbar = tqdm(total=N, desc="prepare", unit="img", dynamic_ncols=True)
    for s in range(0, N, a.batch_size):
        chunk = items[s:s + a.batch_size]
        e = s + len(chunk)
        imgs = torch.stack([load_square(p, IMG_SIZE, a.vcrop) for p, _ in chunk]).to(device)
        with torch.no_grad():
            dist = vae.encode(imgs).latent_dist  # fp32 on purpose
            mean, std = dist.mean * VAE_SCALE, dist.std * VAE_SCALE  # stored already scaled
        texts = [t for _, t in chunk]
        n_trunc += sum(len(tokenizer(t)["input_ids"]) > a.ctx_len for t in texts)
        emb, mask = encode_text(tokenizer, t5, texts, a.ctx_len, device)
        emb16 = emb.half()
        if not torch.isfinite(emb16).all():
            sys.exit("text embeddings overflow fp16; change the text_emb dtype in cmd_prepare to np.float32")
        mean_mm[s:e] = mean.cpu().numpy().astype(f16)
        std_mm[s:e] = std.cpu().numpy().astype(f16)
        emb_mm[s:e] = emb16.cpu().numpy()
        mask_mm[s:e] = mask.cpu().numpy().astype(np.uint8)
        pbar.update(len(chunk))
    pbar.close()

    for m in (mean_mm, std_mm, emb_mm, mask_mm):
        m.flush()
    (out / "meta.json").write_text(json.dumps(
        {"n": N, "ctx_len": a.ctx_len, "files": [str(p) for p, _ in items]}, indent=1))
    if n_trunc:
        log(f"[prepare] WARNING: {n_trunc} captions exceed {a.ctx_len} tokens and were truncated")
    log(f"[prepare] done -> {out}")


# --------------------------------------------------------------------------- train
class CachedDataset(Dataset):
    # The memmaps are opened lazily inside each worker. Holding them as attributes would
    # make Windows ("spawn") workers pickle the whole arrays, which copies them (>1 GB) and fails.
    NAMES = ("latent_mean", "latent_std", "text_emb", "text_mask")

    def __init__(self, cache: Path):
        self.cache = Path(cache)
        self._arrays = None
        emb = np.load(self.cache / "text_emb.npy", mmap_mode="r")
        self.n, self.ctx_len = emb.shape[0], emb.shape[1]
        del emb

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._arrays is None:
            self._arrays = [np.load(self.cache / f"{n}.npy", mmap_mode="r") for n in self.NAMES]
        return tuple(torch.from_numpy(np.array(x[i])) for x in self._arrays)


def infinite(dl):
    while True:
        yield from dl


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1.0 - decay)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def sample_t(B, dist, device):
    if dist == "uniform":
        return torch.rand(B, device=device)
    return torch.sigmoid(torch.randn(B, device=device))  # logit-normal(0, 1)


@torch.no_grad()
def make_previews(ema, prompts, tokenizer, t5, vae, u_txt, u_mask, a, device, path):
    import torchvision.utils as vutils
    n = len(prompts)
    ctx, cmask = encode_text(tokenizer, t5, prompts, u_txt.shape[0], device)
    ctx_all = torch.cat([ctx, u_txt.unsqueeze(0).expand(n, -1, -1)])
    mask_all = torch.cat([cmask.float(), u_mask.unsqueeze(0).expand(n, -1)])
    g = torch.Generator(device=device).manual_seed(a.seed)  # same noise every time
    z = torch.randn(n, LATENT_CH, LATENT_SIZE, LATENT_SIZE, generator=g, device=device)
    dt = 1.0 / a.sample_steps
    for i in range(a.sample_steps):
        t = torch.full((n,), i * dt, device=device)
        with amp(device):
            vb = ema(torch.cat([z, z]), torch.cat([t, t]), ctx_all, mask_all)
        vc, vu = vb.float().chunk(2)
        z = z + dt * (vu + a.sample_cfg * (vc - vu))
    with amp(device):
        imgs = vae.decode(z / VAE_SCALE).sample
    imgs = (imgs.float().clamp(-1, 1) + 1) / 2
    vutils.save_image(imgs, path, nrow=int(math.ceil(math.sqrt(n))))
    log(f"[preview] {path}")


def save_ckpt(path, model, ema, opt, step, cfg, with_opt=True):
    state = {"model": model.state_dict(), "ema": ema.state_dict(), "step": step, "config": cfg}
    if with_opt:
        state["opt"] = opt.state_dict()
    torch.save(state, path)


def cmd_train(a):
    device = pick_device()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = CachedDataset(Path(a.cache))
    bs = min(a.batch_size, len(ds))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=a.workers,
                    pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    log(f"[data] {len(ds)} images, batch {bs} -> {a.steps * bs / len(ds):.1f} epochs over {a.steps} steps")

    base = ensure_checkpoint(a.base_ckpt)
    state = torch.load(base, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        cfg = dict(state.get("config", {}) or {})
        weights = state["ema"] if "ema" in state else state.get("model", state)
    else:
        cfg, weights = {}, state

    model = SupraDiT().to(device)
    model.load_state_dict(weights, strict=True)
    ema = SupraDiT().to(device)
    ema.load_state_dict(weights, strict=True)
    ema.requires_grad_(False).eval()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=a.wd)

    step = 0
    if a.resume:
        r = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        ema.load_state_dict(r["ema"])
        opt.load_state_dict(r["opt"])
        step, cfg = r["step"], r["config"]
        log(f"[resume] from step {step}")

    ctx_len = ds.ctx_len
    tokenizer = t5 = vae = None
    need_t5 = "uncond_text" not in cfg or bool(a.sample_prompts)
    if need_t5:
        tokenizer, t5 = load_t5(device)
    if "uncond_text" not in cfg:  # fall back to encoding the empty string
        e, m = encode_text(tokenizer, t5, [""], ctx_len, device)
        cfg["uncond_text"], cfg["uncond_mask"] = e[0].cpu(), m[0].float().cpu()
        log("[cfg] no stored unconditional embedding in checkpoint; encoded '' instead")
    cfg.setdefault("patch", 2)
    cfg["ctx_len"] = ctx_len
    u_txt = cfg["uncond_text"].to(device).float()
    u_mask = cfg["uncond_mask"].to(device).float()

    prompts = []
    if a.sample_prompts:
        from diffusers import AutoencoderKL
        prompts = [l.strip() for l in Path(a.sample_prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
        vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
        vae.requires_grad_(False)
        if step == 0:  # baseline: what the base model produces, to compare against later
            make_previews(ema, prompts, tokenizer, t5, vae, u_txt, u_mask, a, device, out / "preview_step0.png")

    it = infinite(dl)
    saved = []
    run_loss, t0, smooth = 0.0, time.time(), None
    pbar = tqdm(total=a.steps, initial=step, desc="train", unit="step", dynamic_ncols=True)
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        mean, std, emb, mask = (x.to(device, non_blocking=True) for x in next(it))
        B = mean.shape[0]

        x1 = mean.float() + std.float() * torch.randn(mean.shape, device=device)  # sampled latent
        x0 = torch.randn_like(x1)                                                 # noise
        t = sample_t(B, a.t_dist, device)
        xt = (1 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * x1
        target = x1 - x0

        drop = torch.rand(B, device=device) < a.cond_drop  # caption dropout for CFG
        emb = torch.where(drop[:, None, None], u_txt.unsqueeze(0), emb.float())
        mask = torch.where(drop[:, None], u_mask.unsqueeze(0), mask.float())

        with amp(device):
            pred = model(xt, t, emb, mask)
        loss = F.mse_loss(pred.float(), target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        ema_update(ema, model, a.ema_decay)
        step += 1
        lv = loss.item()
        run_loss += lv
        smooth = lv if smooth is None else 0.98 * smooth + 0.02 * lv
        pbar.update(1)
        pbar.set_postfix(loss=f"{smooth:.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")

        if step % a.log_every == 0:
            log(f"[train] step {step}/{a.steps} loss {run_loss / a.log_every:.4f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {a.log_every / (time.time() - t0):.2f} it/s")
            run_loss, t0 = 0.0, time.time()
        if prompts and step % a.sample_every == 0:
            log(f"[preview] generating samples at step {step} ...")
            make_previews(ema, prompts, tokenizer, t5, vae, u_txt, u_mask, a, device,
                          out / f"preview_step{step}.png")
        if step % a.save_every == 0 and step < a.steps:
            p = out / f"ckpt_step{step}.pt"
            save_ckpt(p, model, ema, opt, step, cfg)
            saved.append(p)
            while len(saved) > a.keep:
                saved.pop(0).unlink(missing_ok=True)

    pbar.close()
    final = out / "model_final_ema.pt"
    save_ckpt(final, model, ema, opt, step, cfg, with_opt=False)
    log(f"[done] {final}  (copy next to inference.py and run it as usual)")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="encode images + captions into a cache")
    p.add_argument("--data", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ctx-len", type=int, default=MAX_CTX_LEN)
    p.add_argument("--min-size", type=int, default=IMG_SIZE,
                   help="skip images whose short side is below this (avoids upscaling blur)")
    p.add_argument("--vcrop", type=float, default=0.35,
                   help="vertical crop anchor for tall images: 0=top, 0.5=center (faces sit high)")
    p.set_defaults(fn=cmd_prepare)

    t = sub.add_parser("train", help="finetune")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base-ckpt", default="./model_final_ema.pt")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--min-lr-ratio", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--ema-decay", type=float, default=0.999,
                   help="use ~0.99-0.995 for runs shorter than ~2000 steps")
    t.add_argument("--cond-drop", type=float, default=0.1, help="caption dropout prob (keeps CFG working)")
    t.add_argument("--t-dist", choices=["logit_normal", "uniform"], default="logit_normal")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=100,
                   help="also print a plain loss line every N steps (kept in the console history)")
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--keep", type=int, default=2, help="how many intermediate checkpoints to keep")
    t.add_argument("--resume", default=None)
    t.add_argument("--sample-prompts", default=None, help="text file, one prompt per line")
    t.add_argument("--sample-every", type=int, default=500)
    t.add_argument("--sample-steps", type=int, default=30)
    t.add_argument("--sample-cfg", type=float, default=3.0)
    t.set_defaults(fn=cmd_train)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()