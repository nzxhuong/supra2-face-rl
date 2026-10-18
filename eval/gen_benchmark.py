#!/usr/bin/env python3
"""
Batch-generate the benchmark set for several Supra2-IMG checkpoints.

Put this next to inference.py. Example:

  # 1) check prompt truncation first (needs no checkpoint)
  python gen_benchmark.py --prompts prompts.txt --check-tokens

  # 2) generate
  python gen_benchmark.py --prompts prompts.txt \
      --ckpt base=./model_final_ema.pt sft=./runs/sft/model_final_ema.pt grpo=./runs/grpo/model_final_ema.pt \
      --seeds 0 1 2 --cfg 3.0 --steps 50 --bs 32 --out bench

Output: bench/<name>/p{prompt_id:04d}_s{seed}.png  (+ bench/prompts.tsv mapping ids to text)
The initial noise depends only on (prompt_id, seed), so every checkpoint starts from identical noise.
Existing files are skipped, so you can stop and resume.
"""
from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

from inference import (LATENT_CH, LATENT_SIZE, MAX_CTX_LEN, T5_NAME, VAE_NAME,
                       VAE_SCALE, SupraDiT, pick_device)


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def check_tokens(prompts, tokenizer, ctx_len):
    lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    over = sum(l > ctx_len for l in lens)
    print(f"tokens: min={min(lens)} mean={sum(lens)/len(lens):.0f} max={max(lens)}")
    print(f"{over}/{len(lens)} prompts exceed ctx_len={ctx_len} and will be truncated")
    ids = tokenizer(prompts[0], truncation=True, max_length=ctx_len)["input_ids"]
    print("\nExample of what the model actually sees for prompt 0:\n"
          + tokenizer.decode(ids, skip_special_tokens=True))


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    weights = state["ema"] if "ema" in state else state.get("model", state)
    model = SupraDiT().to(device).eval()
    model.load_state_dict(weights, strict=True)
    return model, cfg


@torch.no_grad()
def encode(texts, tokenizer, t5, ctx_len, device):
    tok = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=ctx_len, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        ctx = t5(**tok).last_hidden_state.float()
    return ctx, tok["attention_mask"].float()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ckpt", nargs="+", default=[], help="name=path pairs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="only first N prompts (smoke test)")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--check-tokens", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer, T5EncoderModel
    from diffusers import AutoencoderKL
    from torchvision.utils import save_image

    prompts = read_prompts(args.prompts)[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(T5_NAME)
    if args.check_tokens:
        check_tokens(prompts, tokenizer, MAX_CTX_LEN)
        return

    device = pick_device()
    t5 = T5EncoderModel.from_pretrained(T5_NAME).to(device).eval()
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "prompts.tsv"), "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(f"{i}\t{p}\n")

    jobs = [(i, s) for i in range(len(prompts)) for s in args.seeds]

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        model, cfg = load_model(path, device)
        ctx_len = int(cfg.get("ctx_len", MAX_CTX_LEN))
        todo = [(i, s) for i, s in jobs
                if not os.path.isfile(os.path.join(out_dir, f"p{i:04d}_s{s}.png"))]
        print(f"[{name}] {path}  ctx_len={ctx_len}  to generate: {len(todo)}")

        for b in tqdm(range(0, len(todo), args.bs), desc=name):
            batch = todo[b: b + args.bs]
            n = len(batch)
            ctx, mask = encode([prompts[i] for i, _ in batch], tokenizer, t5, ctx_len, device)
            if "uncond_text" in cfg:
                u_ctx = cfg["uncond_text"].to(device).float().unsqueeze(0).expand(n, -1, -1)
                u_mask = cfg["uncond_mask"].to(device).float().unsqueeze(0).expand(n, -1)
            else:
                u_ctx, u_mask = encode([""] * n, tokenizer, t5, ctx_len, device)

            # per-item CPU generator -> identical starting noise across checkpoints
            z = torch.stack([
                torch.randn(LATENT_CH, LATENT_SIZE, LATENT_SIZE,
                            generator=torch.Generator().manual_seed(i * 1000 + s))
                for i, s in batch]).to(device)

            dt = 1.0 / args.steps
            for k in range(args.steps):
                t = torch.full((n,), k * dt, device=device)
                with torch.autocast(**amp):
                    if args.cfg > 1.0:
                        v = model(torch.cat([z, z]), torch.cat([t, t]),
                                  torch.cat([ctx, u_ctx]), torch.cat([mask, u_mask])).float()
                        vc, vu = v.chunk(2)
                        v = vu + args.cfg * (vc - vu)
                    else:
                        v = model(z, t, ctx, mask).float()
                z = z + dt * v

            with torch.autocast(**amp):
                imgs = vae.decode(z / VAE_SCALE).sample
            imgs = ((imgs.float().clamp(-1, 1) + 1) / 2).cpu()
            for img, (i, s) in zip(imgs, batch):
                save_image(img, os.path.join(out_dir, f"p{i:04d}_s{s}.png"))

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
