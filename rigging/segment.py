"""Mark a region named in words in images: SAM 3 (facebook/sam3, gated on Hugging Face) from a text prompt.

python -m rigging.segment PROMPT OUT_PREFIX IMAGE [IMAGE ...] [--threshold 0.4]
    OUT_PREFIX_<i>.png per image (white = the region, the union of every instance found), i counting from 0.
PROMPT is a short noun phrase ("hair", "fur", "mane", "cape"); several separated by commas are joined.
Runs in a Python with transformers >= 5.20 (e.g. ~/venvs/sam3d-xpu), on cuda, xpu or cpu.
"""
import argparse

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.4)
    a = ap.parse_args()
    from transformers import Sam3Model, Sam3Processor

    dev = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float32).to(dev).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    words = [w.strip() for w in a.prompt.split(",") if w.strip()]
    for i, path in enumerate(a.images):
        img = Image.open(path).convert("RGB")
        union = np.zeros((img.height, img.width), bool)
        for w in words:
            inputs = proc(images=img, text=w, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inputs)
            r = proc.post_process_instance_segmentation(out, threshold=a.threshold, mask_threshold=0.5,
                                                        target_sizes=[(img.height, img.width)])[0]
            for m, s in zip(r["masks"], r["scores"]):
                union |= m.cpu().numpy().astype(bool)
            print(f"{path}: '{w}' {len(r['masks'])} instance(s), scores {[round(float(s), 2) for s in r['scores']]}",
                  flush=True)
        Image.fromarray((union * 255).astype(np.uint8)).save(f"{a.out}_{i}.png")


if __name__ == "__main__":
    main()
