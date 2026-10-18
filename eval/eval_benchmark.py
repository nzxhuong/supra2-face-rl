#!/usr/bin/env python3
"""
Evaluate the images made by gen_benchmark.py with ImageReward and/or InsightFace.

  python eval_benchmark.py --bench bench --ckpts base sft grpo                    # both metrics
  python eval_benchmark.py --bench bench --ckpts base sft grpo --metrics ir       # ImageReward only
  python eval_benchmark.py --bench bench --ckpts base sft grpo --metrics face     # InsightFace only


Outputs in <bench>/:
  metrics.csv        one row per (ckpt, prompt, seed)
  summary.md         headline table (95% bootstrap CI over prompts) + per-attribute breakdown
  grid_random.png    same prompt/seed across checkpoints, random prompts
  grid_failures.png  prompts where the LAST checkpoint fails the face check
                     (or, with --metrics ir only, has the lowest ImageReward)
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Prompt parsing: attributes for the breakdown + a short prompt for ImageReward
# ----------------------------------------------------------------------------
def head_clause(p: str) -> str:
    m = re.search(r"portrait of (an? .+?), photographed", p)
    return m.group(1) if m else p[:150]


def short_prompt(p: str) -> str:
    # ImageReward's text encoder (BLIP) only reads ~35 tokens, so give it the subject part
    return "a portrait photo of " + head_clause(p)


def parse_attrs(p: str) -> dict:
    low = p.lower()
    who = head_clause(p).split(",")[0].lower()
    m = re.search(r"(\d+)-year-old", who)
    if m:
        a = int(m.group(1))
        age = "<30" if a < 30 else "30-49" if a < 50 else "50-69" if a < 70 else "70+"
    else:
        age = "70+" if "elderly" in who else "unknown"
    gender = "unknown"
    for g in ["transgender woman", "transgender man", "nonbinary", "androgynous", "woman", "man"]:
        if g in who:
            gender = g
            break
    angle = "other"
    for key, name in [("three-quarter", "three-quarter"), ("straight-on", "straight-on"),
                      ("tilted", "tilted"), ("candid", "candid"),
                      ("off-center", "off-center"), ("third of the frame", "off-center")]:
        if key in low:
            angle = name
            break
    lm = re.search(r"lighting: (.+?)\.", low)
    light = lm.group(1) if lm else ""
    lighting = "dramatic/low-key" if re.search(r"low-key|rembrandt|moody|candle|hard sun|dramatic", light) else "soft/other"
    return dict(
        age=age, gender=gender, angle=angle, lighting=lighting,
        glasses=bool(re.search(r"glasses|sunglasses|eyeglasses|spectacles", low)),
        headwear=bool(re.search(r"\b(hat|cap|beanie|headband|hood|turban|beret|bandana)\b", low)),
    )


# ----------------------------------------------------------------------------
# Scorers (each imported lazily, only if requested)
# ----------------------------------------------------------------------------
def build_scorers(metrics):
    import torch

    app = ir = None
    if "face" in metrics:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"])
        app.prepare(ctx_id=0 if cuda else -1, det_size=(512, 512))
    if "ir" in metrics:
        ir = load_imagereward("cuda" if torch.cuda.is_available() else "cpu")
    return app, ir


def load_imagereward(device):
    """ImageReward's package __init__ imports ReFL (training code), which drags in `datasets`
    and `diffusers`. The scorer itself needs neither, so if the normal import fails we create
    the package shell ourselves and import only ImageReward.utils."""
    import importlib.util
    import sys
    import types

    try:
        import ImageReward as RM
        return RM.load("ImageReward-v1.0", device=device)
    except Exception as first_error:  # noqa: BLE001
        print(f"[ImageReward] normal import failed ({first_error!r}); loading without ReFL")

    for k in [k for k in sys.modules if k == "ImageReward" or k.startswith("ImageReward.")]:
        del sys.modules[k]
    spec = importlib.util.find_spec("ImageReward")
    if spec is None:
        raise SystemExit("image-reward is not installed in this env: pip install image-reward")
    pkg = types.ModuleType("ImageReward")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__spec__ = spec
    sys.modules["ImageReward"] = pkg
    try:
        from ImageReward.utils import load
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not import ImageReward: {e!r}\n"
            "Check:  pip install \"transformers==4.44.2\" \"huggingface_hub==0.25.2\" fairscale timm ftfy regex\n"
            "        pip install git+https://github.com/openai/CLIP.git")
    return load("ImageReward-v1.0", device=device)


def face_stats(app, path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    big = img.resize((w * 2, h * 2), Image.LANCZOS)  # detectors are happier above 256px; same for every ckpt
    faces = app.get(np.array(big)[:, :, ::-1])
    if not faces:
        return dict(n_faces=0, det_score=0.0, face_area=0.0)
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) / (w * 2 * h * 2)
    return dict(n_faces=len(faces), det_score=float(f.det_score), face_area=float(area))


def score_all(args, prompts):
    csv_path = os.path.join(args.bench, "metrics.csv")
    rows = {}
    if os.path.isfile(csv_path):
        for r in pd.read_csv(csv_path).to_dict("records"):
            rows[(r["ckpt"], int(r["pid"]), int(r["seed"]))] = r

    need = []  # (key, path, need_face, need_ir)
    for name in args.ckpts:
        d = os.path.join(args.bench, name)
        for fn in sorted(os.listdir(d)):
            m = re.match(r"p(\d+)_s(\d+)\.png$", fn)
            if not m:
                continue
            key = (name, int(m[1]), int(m[2]))
            r = rows.get(key, {})
            mf = "face" in args.metrics and pd.isna(r.get("n_faces"))
            mi = "ir" in args.metrics and pd.isna(r.get("ir"))
            if mf or mi:
                need.append((key, os.path.join(d, fn), mf, mi))

    if need:
        app, ir = build_scorers(args.metrics)
        for key, path, mf, mi in tqdm(need, desc="scoring"):
            r = rows.get(key, dict(ckpt=key[0], pid=key[1], seed=key[2]))
            if mf:
                r.update(face_stats(app, path))
            if mi:
                r["ir"] = float(ir.score(short_prompt(prompts[key[1]]), path))
            rows[key] = r

    df = pd.DataFrame(list(rows.values()))
    df.to_csv(csv_path, index=False)
    return df[df.ckpt.isin(args.ckpts)].copy()


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def bootstrap(df, col, ckpts, nboot=2000, seed=0):
    """Per-prompt means, resampled over prompts (seeds of one prompt are not independent)."""
    arr = df.groupby(["pid", "ckpt"])[col].mean().unstack("ckpt")[ckpts].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(nboot, len(arr)))
    means = arr[idx].mean(1)  # (nboot, C)
    return arr.mean(0), np.percentile(means, [2.5, 97.5], axis=0), means


def fmt(m, lo, hi, pct=False):
    return f"{m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]" if pct else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def write_summary(df, prompts, args):
    ck = args.ckpts
    has_face = "n_faces" in df and df["n_faces"].notna().all()
    has_ir = "ir" in df and df["ir"].notna().all()
    if not (has_face or has_ir):
        raise SystemExit("No complete metric columns found in metrics.csv.")

    out = [f"# Benchmark summary\n\n{df.pid.nunique()} prompts x {df.seed.nunique()} seeds per checkpoint. "
           "Brackets are 95% bootstrap CIs over prompts."]
    specs = []
    if has_face:
        df["one_face"] = (df.n_faces == 1).astype(float)
        df["no_face"] = (df.n_faces == 0).astype(float)
        df["multi_face"] = (df.n_faces > 1).astype(float)
        df["clean"] = ((df.n_faces == 1) & (df.det_score >= args.min_det) & (df.face_area >= args.min_area)).astype(float)
        out.append(f"'Clean' = exactly one face, det score >= {args.min_det}, "
                   f"face area >= {args.min_area:.0%} of the image.")
        specs += [("clean", "clean-face rate", True), ("one_face", "exactly one face", True),
                  ("no_face", "no face", True), ("multi_face", "multiple faces", True),
                  ("det_score", "mean det score", False)]
    if has_ir:
        specs.append(("ir", "ImageReward", False))

    table = ["| metric | " + " | ".join(ck) + " |", "|---|" + "---|" * len(ck)]
    means = {}
    for col, label, pct in specs:
        m, ci, bs = bootstrap(df, col, ck)
        means[col] = bs
        table.append(f"| {label} | " + " | ".join(fmt(m[i], ci[0, i], ci[1, i], pct) for i in range(len(ck))) + " |")
    out += ["\n" + "\n".join(table), "\n## Paired differences (next checkpoint minus previous)\n"]
    for col, pct in [("clean", True), ("ir", False)]:
        if col not in means:
            continue
        for i in range(1, len(ck)):
            d = means[col][:, i] - means[col][:, i - 1]
            lo, hi = np.percentile(d, [2.5, 97.5])
            s = (f"{d.mean()*100:+.1f} pts [{lo*100:.1f}, {hi*100:.1f}]" if pct
                 else f"{d.mean():+.3f} [{lo:.3f}, {hi:.3f}]")
            out.append(f"- {col}: {ck[i]} - {ck[i-1]} = {s}")

    if has_ir:
        out.append("\n## ImageReward win rate (same prompt + seed)\n")
        piv = df.pivot_table(index=["pid", "seed"], columns="ckpt", values="ir")[ck].dropna()
        for i in range(1, len(ck)):
            out.append(f"- {ck[i]} beats {ck[i-1]}: {(piv[ck[i]] > piv[ck[i-1]]).mean():.1%} of {len(piv)} pairs")

    # attribute breakdown (clean-face rate if available, otherwise mean ImageReward)
    prim = "clean" if has_face else "ir"
    attrs = pd.DataFrame([parse_attrs(p) for p in prompts]).rename_axis("pid").reset_index()
    dfa = df.merge(attrs, on="pid")
    out.append(f"\n## {'Clean-face rate' if prim == 'clean' else 'Mean ImageReward'} by attribute "
               "(groups with >= 10 prompts)\n")
    for a in ["age", "gender", "glasses", "headwear", "angle", "lighting"]:
        g = dfa.groupby([a, "ckpt"])[prim].mean().unstack("ckpt")[ck]
        n = attrs.groupby(a).size()
        g = g[n.reindex(g.index) >= 10]
        if g.empty:
            continue
        g.insert(0, "n_prompts", n.reindex(g.index))
        out.append(f"**{a}**\n\n" + (g.round(3).to_markdown() if _has_tabulate() else g.round(3).to_string()) + "\n")

    with open(os.path.join(args.bench, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    return df


# ----------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------
def make_grid(bench, ckpts, pids, seed, path, size=256):
    if not pids:
        return
    lab, pad = 22, 4
    W = len(ckpts) * (size + pad) + 60
    H = lab + len(pids) * (size + pad)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for c, name in enumerate(ckpts):
        d.text((60 + c * (size + pad) + 4, 4), name, fill="black")
    for r, pid in enumerate(pids):
        y = lab + r * (size + pad)
        d.text((4, y + size // 2), f"#{pid}", fill="black")
        for c, name in enumerate(ckpts):
            fp = os.path.join(bench, name, f"p{pid:04d}_s{seed}.png")
            if os.path.isfile(fp):
                canvas.paste(Image.open(fp).convert("RGB").resize((size, size)), (60 + c * (size + pad), y))
    canvas.save(path)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench")
    ap.add_argument("--ckpts", nargs="+", required=True, help="in order, e.g. base sft grpo")
    ap.add_argument("--metrics", nargs="+", choices=["face", "ir"], default=["face", "ir"])
    ap.add_argument("--min-det", type=float, default=0.7)
    ap.add_argument("--min-area", type=float, default=0.03)
    ap.add_argument("--grid-rows", type=int, default=8)
    ap.add_argument("--grid-seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.bench, "prompts.tsv"), encoding="utf-8") as f:
        prompts = [l.rstrip("\n").split("\t", 1)[1] for l in f if l.strip()]

    df = score_all(args, prompts)
    df = write_summary(df, prompts, args)

    rng = np.random.default_rng(0)
    pids = sorted(rng.choice(len(prompts), size=min(args.grid_rows, len(prompts)), replace=False).tolist())
    make_grid(args.bench, args.ckpts, pids, args.grid_seed, os.path.join(args.bench, "grid_random.png"))

    last = df[(df.ckpt == args.ckpts[-1]) & (df.seed == args.grid_seed)]
    bad = last[last.clean == 0] if "clean" in last else last.sort_values("ir")
    make_grid(args.bench, args.ckpts, bad.pid.head(args.grid_rows).tolist(), args.grid_seed,
              os.path.join(args.bench, "grid_failures.png"))


if __name__ == "__main__":
    main()