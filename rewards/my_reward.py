import sys, glob, numpy as np, mediapipe as mp
from PIL import Image
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path="face_landmarker.task"), num_faces=2))

_st = None

def feats(im):
    im = im.convert("RGB").resize((256, 256), Image.BICUBIC)
    arr = np.ascontiguousarray(np.asarray(im.convert("RGB").resize((512, 512), Image.BICUBIC)))
    r = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if len(r.face_landmarks) != 1:
        return None
    p = np.array([[l.x, l.y] for l in r.face_landmarks[0]])
    d = lambda a, b: np.linalg.norm(p[a] - p[b])
    fw, fh = d(234, 454), d(10, 152)
    em, mm = (p[33] + p[263]) / 2, (p[61] + p[291]) / 2
    return np.array([d(33, 263) / fw, np.linalg.norm(em - mm) / fh, d(61, 291) / fw,
                     fh / fw, abs(p[1, 0] - (p[234, 0] + p[454, 0]) / 2) / fw,
                     abs(p[33, 1] - p[263, 1]) / fw, fh])

def my_reward(ctx):
    global _st
    if _st is None: _st = np.load("face_stats.npz")
    out = []
    for im in ctx.images:
        f = feats(im)
        out.append(-3.0 if f is None else
                   -float(np.clip(np.abs((f - _st["mu"]) / _st["sd"]) - 1.0, 0, 3).mean()))
    return out

if __name__ == "__main__": 
    F = [f for f in (feats(Image.open(p)) for p in glob.glob(sys.argv[1] + "/*.*")[:3000]) if f is not None]
    F = np.stack(F); np.savez("face_stats.npz", mu=F.mean(0), sd=F.std(0)); print(len(F), F.mean(0), F.std(0))