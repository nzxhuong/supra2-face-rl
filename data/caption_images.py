#!/usr/bin/env python3
"""
Caption a folder of portrait images with a vision-language model.

For every image `x.jpg` this writes `x.txt` next to it (the layout that
supra2_finetune.py expects). Images the model judges unsuitable (blurry, filtered,
watermarked, no clear face, ...) get no .txt; they are listed in rejects.tsv instead.
Re-running skips images that already have a caption or a reject entry.

    pip install -U transformers accelerate pillow torch
    python caption_images.py --images ./portraits

Any instruction-tuned VLM that transformers can load through
AutoModelForImageTextToText works; pass it with --model. The prompt lives in
caption_prompt.txt, so you can also paste it into any hosted VLM API instead.
"""
import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clean(text: str) -> str:
    text = " ".join(text.split())
    return text.strip().strip("\"'`").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with images (searched recursively)")
    ap.add_argument("--prompt-file", default=str(Path(__file__).with_name("caption_prompt.txt")))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="downscale the longest side before captioning (limits VLM tokens)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--rejects", default=None, help="default: <images>/rejects.tsv")
    a = ap.parse_args()


    root = Path(a.images)
    prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip()
    rejects_path = Path(a.rejects) if a.rejects else root / "rejects.tsv"
    rejected = set()
    if rejects_path.exists():
        rejected = {l.split("\t")[0] for l in rejects_path.read_text(encoding="utf-8").splitlines() if l}

    todo = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if p.with_suffix(".txt").exists() or str(p) in rejected:
            continue
        todo.append(p)
    print(f"{len(todo)} images to caption", flush=True)
    if not todo:
        return

    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    n_ok = n_rej = n_long = 0
    with open(rejects_path, "a", encoding="utf-8") as rej_f:
        for i, p in enumerate(todo, 1):
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:  # unreadable file
                rej_f.write(f"{p}\tunreadable: {e}\n")
                rej_f.flush()
                n_rej += 1
                continue
            s = a.max_side / max(img.size)
            if s < 1:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            text = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            text = clean(text)

            if text.upper().startswith("REJECT"):
                reason = re.sub(r"^REJECT:?\s*", "", text, flags=re.I)
                rej_f.write(f"{p}\t{reason}\n")
                rej_f.flush()
                n_rej += 1
            else:
                p.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
                n_ok += 1
                if len(text.split()) > 90:
                    n_long += 1
            if i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] captioned={n_ok} rejected={n_rej}", flush=True)

    print(f"done. captioned={n_ok} rejected={n_rej} (see {rejects_path})")
    if n_long:
        print(f"note: {n_long} captions are over 90 words; Flan-T5 context is 128 tokens, "
              f"so very long captions get truncated at training time.", file=sys.stderr)


if __name__ == "__main__":
    main()
