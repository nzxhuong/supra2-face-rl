import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from my_reward import my_reward   # MediaPipe reward

name = "HuggingFaceTB/SmolVLM-Instruct"
dev = 1 if torch.cuda.device_count() > 1 else 0
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
proc = AutoProcessor.from_pretrained(name, do_image_splitting=False)
proc.tokenizer.padding_side = "left"
vlm = AutoModelForImageTextToText.from_pretrained(name, quantization_config=cfg, device_map={"": dev}).eval()
tok = proc.tokenizer
yes = [tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes"]]
no = [tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No"]]
qs = ["Is there a human face in this image?",
      "Is this a clear, well-formed human face?",
      "Does this show a person's head and shoulders, centered?"]

@torch.no_grad()
def vlm_score(images, q, bs=16):
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q + " Answer Yes or No."}]}]
    prompt = proc.apply_chat_template(msg, add_generation_prompt=True)
    out = []
    for i in range(0, len(images), bs):
        ims = images[i:i + bs]
        inp = proc(text=[prompt] * len(ims), images=[[im.convert("RGB")] for im in ims],
                   return_tensors="pt", padding=True).to(vlm.device)
        lg = vlm(**inp).logits[:, -1].float()
        out.append(torch.sigmoid(torch.logsumexp(lg[:, yes], -1) - torch.logsumexp(lg[:, no], -1)).cpu())
    return torch.cat(out)

W_MP = 0.5

def combo_reward(ctx):
    v = sum(vlm_score(ctx.images, q) for q in qs) / len(qs)       # [0,1]
    mp = torch.tensor(my_reward(ctx))                             # [-3,0]
    mp = torch.where(mp <= -3.0, torch.zeros_like(mp), 1 + mp / 3)  # no face -> 0
    return (v + W_MP * mp).tolist()