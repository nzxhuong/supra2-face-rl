#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from inference import ( 
        LATENT_CH, LATENT_SIZE, MAX_CTX_LEN, T5_NAME, VAE_NAME, VAE_SCALE,
        SupraDiT, ensure_checkpoint, pick_device,
    )
except ImportError:
    sys.exit("inference.py not found next to this script.")

try:
    from diffusion_tinker.core.ema import ParameterEMA
    from diffusion_tinker.core.noise_strategy import sde_diffusion_std, sde_step_with_logprob
    from diffusion_tinker.core.stat_tracking import PerPromptStatTracker
    from diffusion_tinker.rewards.protocol import RewardContext
    from diffusion_tinker.rewards.resolve import resolve_reward
except ImportError:
    sys.exit("diffusion-tinker not installed:\n"
             "  pip install --no-deps git+https://github.com/srijitiyer/diffusion-tinker.git")


def log(msg: str) -> None:
    tqdm.write(msg)


def amp(device: torch.device):
    # cache_enabled=False: rollout and replay must run the exact same numerics, otherwise the
    # importance ratio is biased away from 1 (see diffusion-tinker's sd3_patch.py comments).
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    return contextlib.nullcontext()


# =============================================================================== text / images
@torch.no_grad()
def make_text_encoder(device, ctx_len):
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(T5_NAME)
    t5 = T5EncoderModel.from_pretrained(T5_NAME).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(prompts: list[str]):
        t = tok(prompts, padding="max_length", truncation=True, max_length=ctx_len,
                return_tensors="pt").to(device)
        with amp(device):
            emb = t5(**t).last_hidden_state
        return emb.float(), t["attention_mask"].float()

    return encode


def make_vae_decoder(device, chunk=8):
    from diffusers import AutoencoderKL
    from PIL import Image
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def decode(z: torch.Tensor):
        """latents (B,4,32,32) -> list[PIL.Image]"""
        out = []
        for i in range(0, z.shape[0], chunk):
            with amp(device):
                x = vae.decode(z[i:i + chunk] / VAE_SCALE).sample
            x = ((x.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            out += [Image.fromarray(a) for a in x.permute(0, 2, 3, 1).cpu().numpy()]
        return out

    return decode


# =============================================================================== reward loading
def load_reward(spec: str, device: str, kwargs: dict):
    """'my_file.py:fn' -> your function; anything else -> a diffusion-tinker built-in
    ('aesthetic', 'clip_score', 'pickscore', 'ocr', ...)."""
    if ":" in spec and spec.split(":")[0].endswith(".py"):
        path, fn_name = spec.rsplit(":", 1)
        mod_spec = importlib.util.spec_from_file_location("user_reward", path)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.path.insert(0, str(Path(path).resolve().parent))
        mod_spec.loader.exec_module(mod)
        fn = getattr(mod, fn_name)
    else:
        fn = spec
    return resolve_reward(fn, device=device), kwargs


# =============================================================================== policy math
def predict_velocity(model, z, sigma, ctx, cmask, u_ctx, u_mask, cfg_scale):
    """CFG-combined model output in diffusion-tinker's convention (noise - data).

    sigma: scalar tensor. The model takes t = 1 - sigma in [0,1] (it scales by 1000 itself).
    """
    B = z.shape[0]
    if cfg_scale > 1.0:
        t = (1.0 - sigma).expand(2 * B)
        v = model(torch.cat([z, z]), t,
                  torch.cat([ctx, u_ctx.expand(B, -1, -1)]),
                  torch.cat([cmask, u_mask.expand(B, -1)]))
        vc, vu = v.float().chunk(2)
        v = vu + cfg_scale * (vc - vu)
    else:
        v = model(z, (1.0 - sigma).expand(B), ctx, cmask).float()
    return -v  # SupraDiT predicts data - noise; the SDE code wants noise - data


class FlowGRPO:
    def __init__(self, a, device, model, base_cfg, encode, decode, reward_fn, reward_meta,
                 train_prompts, eval_prompts):
        self.a, self.device = a, device
        self.encode, self.decode = encode, decode
        self.reward_fn, self.reward_meta = reward_fn, reward_meta
        self.train_prompts, self.eval_prompts = train_prompts, eval_prompts
        self.base_cfg = base_cfg
        self.u_ctx = base_cfg["uncond_text"].to(device).float().unsqueeze(0)  # (1, L, D)
        self.u_mask = base_cfg["uncond_mask"].to(device).float().unsqueeze(0)  # (1, L)
        self.use_lora = a.lora_rank > 0

        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            # names inside SupraDiT: self_attn.qkv / .proj, cross_attn.q / .kv / .proj,
            # mlp.0 / mlp.2, adaln.1 (peft matches on the name suffix)
            targets = ["qkv", "q", "kv", "proj", "mlp.0", "mlp.2"]
            self.model = get_peft_model(model, LoraConfig(
                r=a.lora_rank, lora_alpha=a.lora_alpha, target_modules=targets,
                init_lora_weights="gaussian"))
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            self.ref = None  # reference = adapters disabled
        else:
            self.model = model
            self.ref = None
            if a.kl_beta > 0:
                self.ref = copy.deepcopy(model).to(device).eval().requires_grad_(False)
        self.model.to(device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=a.lr, weight_decay=a.wd)
        n_tr = sum(p.numel() for p in self.params)
        n_all = sum(p.numel() for p in self.model.parameters())
        log(f"[model] trainable {n_tr/1e6:.2f}M / {n_all/1e6:.2f}M params "
            f"({'LoRA r=%d' % a.lora_rank if self.use_lora else 'full fine-tune'})")

        self.ema = ParameterEMA(self.params, decay=a.ema_decay, update_step_interval=a.ema_interval) \
            if a.ema_decay > 0 else None
        self.tracker = PerPromptStatTracker(global_std=a.global_std)
        self.sigmas = torch.linspace(1.0, 0.0, a.steps + 1, device=device)  # = 1 - i/steps (uniform t)
        self.best, self.bad_evals, self.epoch = -float("inf"), 0, 0
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def rollout(self, prompts, noise_level, cfg_scale, steps, init=None, generator=None):
        """Euler-Maruyama rollout. Returns latents, next_latents, log_probs and final latent."""
        self.model.eval()
        B = len(prompts)
        ctx, cmask = self.encode(prompts)
        z = init if init is not None else torch.randn(
            B, LATENT_CH, LATENT_SIZE, LATENT_SIZE, device=self.device, generator=generator)
        sig = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        zs, zns, lps = [], [], []
        for i in range(steps):
            s, sn = sig[i], sig[i + 1]
            with amp(self.device):
                v = predict_velocity(self.model, z, s, ctx, cmask, self.u_ctx, self.u_mask, cfg_scale)
            nl = noise_level if i < steps - 1 else 0.0  # last step lands on sigma=0 exactly
            z_next, lp, _ = sde_step_with_logprob(v, s.expand(B), sn.expand(B), z, nl, generator=generator)
            zs.append(z); zns.append(z_next); lps.append(lp)
            z = z_next
        return dict(latents=torch.stack(zs, 1), next_latents=torch.stack(zns, 1),
                    log_probs=torch.stack(lps, 1), final=z, ctx=ctx, cmask=cmask)

    def score(self, images, prompts, latents=None, epoch=None):
        ctx = RewardContext(images=images, prompts=prompts, device=torch.device(self.device),
                            metadata=dict(self.reward_meta), latents=latents, epoch=epoch)
        s = self.reward_fn(ctx).scores.float().cpu()
        return torch.nan_to_num(s, nan=0.0)

    # ------------------------------------------------------------------ replay
    def replay(self, z_t, z_next, s, sn, ctx, cmask, noise_level, with_ref):
        B = z_t.shape[0]
        def fwd(model):
            with amp(self.device):
                v = predict_velocity(model, z_t, s, ctx, cmask, self.u_ctx, self.u_mask, self.a.cfg)
            return sde_step_with_logprob(v, s.expand(B), sn.expand(B), z_t, noise_level, prev_sample=z_next)
        _, lp, mean = fwd(self.model)
        mean_ref = None
        if with_ref:
            with torch.no_grad():
                if self.use_lora:
                    with self.model.disable_adapter():
                        _, _, mean_ref = fwd(self.model)
                else:
                    _, _, mean_ref = fwd(self.ref)
        return lp, mean, mean_ref

    # ------------------------------------------------------------------ one epoch
    def train_epoch(self, epoch):
        a = self.a
        prompts = random.sample(self.train_prompts, min(a.prompts_per_epoch, len(self.train_prompts)))
        prompts = [p for p in prompts for _ in range(a.group_size)]
        traj = self.rollout(prompts, a.noise_level, a.cfg, a.steps)
        images = self.decode(traj["final"])
        rewards = self.score(images, prompts, traj["final"], epoch)
        adv = self.tracker.update(prompts, rewards).clamp(-a.adv_clip, a.adv_clip).to(self.device)
        if (adv.abs() > 1e-8).sum() < 2:
            return dict(reward=rewards.mean().item(), skipped=1.0)

        T = a.steps
        n_train = max(1, int(T * a.timestep_fraction))
        steps_idx = [j for j in range(min(n_train, T - 1))]  # step T-1 is deterministic: no log-prob
        if a.num_train_timesteps and a.num_train_timesteps < len(steps_idx):
            steps_idx = sorted(random.sample(steps_idx, a.num_train_timesteps))

        self.model.train()
        N = len(prompts)
        perm = torch.randperm(N, device=self.device)
        stats = dict(rl=0.0, kl=0.0, ratio=0.0, clipfrac=0.0, n=0, gnorm=0.0, nopt=0)
        for mb in perm.split(a.train_batch_size):
            self.opt.zero_grad(set_to_none=True)
            order = steps_idx.copy()
            random.shuffle(order)
            for j in order:
                s, sn = self.sigmas[j], self.sigmas[j + 1]
                nl = a.noise_level
                lp, mean, mean_ref = self.replay(
                    traj["latents"][mb, j], traj["next_latents"][mb, j], s, sn,
                    traj["ctx"][mb], traj["cmask"][mb], nl, with_ref=a.kl_beta > 0)
                ratio = torch.exp(lp.float() - traj["log_probs"][mb, j].float())
                A = adv[mb]
                loss_rl = torch.maximum(-A * ratio, -A * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                loss = loss_rl
                kl = torch.zeros((), device=self.device)
                if mean_ref is not None:
                    dt = (sn - s).float()
                    std = sde_diffusion_std(s.float(), sn.float(), nl)
                    var = (std * torch.sqrt((-dt).clamp(min=1e-12))) ** 2
                    kl = ((mean.float() - mean_ref.float()) ** 2).flatten(1).mean(1).div(2 * var + 1e-12).mean()
                    loss = loss + a.kl_beta * kl
                (loss / len(order)).backward()
                stats["rl"] += loss_rl.item(); stats["kl"] += kl.item()
                stats["ratio"] += ratio.mean().item()
                stats["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                stats["n"] += 1
            g = torch.nn.utils.clip_grad_norm_(self.params, a.max_grad_norm)
            self.opt.step()
            stats["gnorm"] += float(g); stats["nopt"] += 1
        n = max(stats["n"], 1)
        return dict(reward=rewards.mean().item(), reward_std=rewards.std().item(),
                    rl_loss=stats["rl"] / n, kl=stats["kl"] / n, mean_ratio=stats["ratio"] / n,
                    clipfrac=stats["clipfrac"] / n, gnorm=stats["gnorm"] / max(stats["nopt"], 1))

    # ------------------------------------------------------------------ eval / export
    @torch.no_grad()
    def evaluate(self, epoch):
        a = self.a
        ps = [p for p in self.eval_prompts for _ in range(a.eval_samples)]
        g = torch.Generator(device=self.device).manual_seed(a.eval_seed)
        init = torch.randn(len(ps), LATENT_CH, LATENT_SIZE, LATENT_SIZE, device=self.device, generator=g)
        if self.ema is not None:
            self.ema.swap_in(self.params)
        try:
            imgs, finals = [], []
            for i in range(0, len(ps), a.train_batch_size):
                o = self.rollout(ps[i:i + a.train_batch_size], 0.0, a.eval_cfg, a.eval_steps,
                                 init=init[i:i + a.train_batch_size])
                imgs += self.decode(o["final"]); finals.append(o["final"])
        finally:
            if self.ema is not None:
                self.ema.swap_out(self.params)
        sc = self.score(imgs, ps, torch.cat(finals), epoch)
        m, sem = sc.mean().item(), sc.std().item() / max(len(sc) ** 0.5, 1)
        d = self.out / f"eval-{epoch}"
        d.mkdir(exist_ok=True)
        for k, im in enumerate(imgs[:16]):
            im.save(d / f"{k}.png")
        log(f"[eval] epoch {epoch}: reward {m:.4f} +/- {sem:.4f} (n={len(sc)})")
        if m > self.best:
            self.best, self.bad_evals = m, 0
            self.export(self.out / "model_best_ema.pt", epoch)
        else:
            self.bad_evals += 1
        return m

    @torch.no_grad()
    def export(self, path, step):
        """Write an inference.py-compatible checkpoint (LoRA merged, EMA weights)."""
        if self.ema is not None:
            self.ema.swap_in(self.params)
        try:
            if self.use_lora:
                merged = copy.deepcopy(self.model).merge_and_unload()
                sd = {k: v.detach().cpu() for k, v in merged.state_dict().items()}
                del merged
            else:
                sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        finally:
            if self.ema is not None:
                self.ema.swap_out(self.params)
        torch.save({"ema": sd, "step": step, "config": self.base_cfg}, path)
        log(f"[save] {path}")

    def run(self):
        a = self.a
        for epoch in range(a.epochs):
            self.epoch = epoch
            if epoch % a.eval_every == 0:
                self.evaluate(epoch)
                if a.patience and self.bad_evals >= a.patience:
                    log(f"[stop] no eval improvement for {a.patience} evals")
                    break
            t0 = time.time()
            m = self.train_epoch(epoch)
            if self.ema is not None and "skipped" not in m:
                self.ema.step(self.params, epoch)
            log(f"[train] epoch {epoch} " + " ".join(f"{k}={v:.4g}" for k, v in m.items())
                + f" ({time.time() - t0:.0f}s)")
        else:
            self.evaluate(a.epochs)
        self.export(self.out / "model_final_ema.pt", self.epoch)


# =============================================================================== cli
def read_lines(p):
    return [l.strip() for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", default="./model_final_ema.pt")
    ap.add_argument("--prompts", required=True, help="text file, one training prompt per line")
    ap.add_argument("--eval-prompts", default=None, help="held-out prompts; default: last --n-eval of --prompts")
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--reward", required=True, help="'file.py:function' or a built-in name (aesthetic, clip_score, ...)")
    ap.add_argument("--reward-kwargs", default="{}", help="JSON -> ctx.metadata in your reward")
    ap.add_argument("--reward-device", default=None, help="e.g. cpu if the reward model does not fit next to the DiT")
    ap.add_argument("--out", default="./runs/grpo")
    ap.add_argument("--epochs", type=int, default=100)
    # rollout
    ap.add_argument("--prompts-per-epoch", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8, help="samples per prompt (>=2)")
    ap.add_argument("--steps", type=int, default=20, help="denoising steps in the training rollout")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--noise-level", type=float, default=0.7)
    # optimisation
    ap.add_argument("--lora-rank", type=int, default=32, help="0 = full fine-tune")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="1e-4 for LoRA; use ~1e-5 for full fine-tune")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--train-batch-size", type=int, default=16, help="replay minibatch; one optimizer step each")
    ap.add_argument("--clip", type=float, default=1e-3, help="PPO clip (log-prob is mean-reduced; watch clipfrac)")
    ap.add_argument("--adv-clip", type=float, default=5.0)
    ap.add_argument("--global-std", action="store_true", help="normalise advantages by batch std (low-variance rewards)")
    ap.add_argument("--timestep-fraction", type=float, default=0.99)
    ap.add_argument("--num-train-timesteps", type=int, default=None, help="random subset of steps per epoch (faster)")
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.9, help="0 = off")
    ap.add_argument("--ema-interval", type=int, default=1)
    # eval
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-steps", type=int, default=30)
    ap.add_argument("--eval-cfg", type=float, default=3.0)
    ap.add_argument("--eval-samples", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.group_size < 2:
        sys.exit("--group-size must be >= 2 (a group of one has zero advantage)")

    device = pick_device()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    base = ensure_checkpoint(a.base_ckpt)
    state = torch.load(base, map_location="cpu", weights_only=False)
    cfg = dict(state.get("config", {}) or {}) if isinstance(state, dict) else {}
    weights = (state["ema"] if "ema" in state else state.get("model", state)) if isinstance(state, dict) else state
    model = SupraDiT()
    model.load_state_dict(weights, strict=True)
    del state, weights

    ctx_len = int(cfg.get("ctx_len", MAX_CTX_LEN))
    encode = make_text_encoder(device, ctx_len)
    if "uncond_text" not in cfg:  # same fallback as supra2_finetune.py
        e, m = encode([""])
        cfg["uncond_text"], cfg["uncond_mask"] = e[0].cpu(), m[0].cpu()
    cfg.setdefault("patch", 2)
    cfg["ctx_len"] = ctx_len
    decode = make_vae_decoder(device)

    reward_fn, meta = load_reward(a.reward, a.reward_device or str(device), json.loads(a.reward_kwargs))

    train_prompts = read_lines(a.prompts)
    if a.eval_prompts:
        eval_prompts = read_lines(a.eval_prompts)
    elif len(train_prompts) > 2 * a.n_eval:
        random.Random(0).shuffle(train_prompts)
        eval_prompts, train_prompts = train_prompts[:a.n_eval], train_prompts[a.n_eval:]
        log(f"[data] held out {a.n_eval} prompts for eval")
    else:
        eval_prompts = train_prompts[:a.n_eval]
        log("[data] WARNING: too few prompts to hold any out; eval reward is measured on training prompts")
    log(f"[data] {len(train_prompts)} train / {len(eval_prompts)} eval prompts")

    FlowGRPO(a, device, model, cfg, encode, decode, reward_fn, meta, train_prompts, eval_prompts).run()


if __name__ == "__main__":
    main()
