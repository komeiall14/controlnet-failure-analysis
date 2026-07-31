"""生成済み画像に CLIP スコア（プロンプト追従）を後付けで与える。

構造整合（edge F1）だけを見ると「条件を強くするほど良い」という結論になりがちだが、
条件を強めると生成はプロンプトより輪郭に従うようになる。両方を測って初めて
「条件という資源を増やすほど良いのか」を検証できる。

拡散は回さないので1枚あたり数十msで済む。
"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IMG = os.path.join(OUT, "images")

_m = None


def get_clip():
    global _m
    if _m is None:
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        _m = (CLIPModel.from_pretrained(name).to("mps").eval(),
              CLIPProcessor.from_pretrained(name))
    return _m


@torch.no_grad()
def score(paths, prompts):
    model, proc = get_clip()
    out = []
    for p, txt in zip(paths, prompts):
        im = Image.open(p).convert("RGB")
        inp = proc(text=[txt], images=im, return_tensors="pt",
                   padding=True, truncation=True).to("mps")
        f_i = model.get_image_features(pixel_values=inp["pixel_values"])
        f_t = model.get_text_features(input_ids=inp["input_ids"],
                                      attention_mask=inp["attention_mask"])
        f_i = f_i / f_i.norm(dim=-1, keepdim=True)
        f_t = f_t / f_t.norm(dim=-1, keepdim=True)
        out.append(float((f_i @ f_t.T).item()))
    return out


def prompt_of(source):
    import dataset as ds
    table = {n: p for n, _, p in ds.real_images() + ds.synthetic_images()}
    return table.get(source, "a photo")


def annotate(csv_name, tag_col="tag"):
    path = os.path.join(OUT, f"{csv_name}.csv")
    if not os.path.exists(path):
        print(f"  skip {csv_name} (未生成)")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    paths, prompts, keep = [], [], []
    for r in rows:
        f = os.path.join(IMG, f"{r[tag_col]}_gen.png")
        if os.path.exists(f):
            paths.append(f)
            prompts.append(prompt_of(r.get("source", "")))
            keep.append(r)
    if not paths:
        print(f"  skip {csv_name} (画像なし)")
        return
    sc = score(paths, prompts)
    for r, s in zip(keep, sc):
        r["clip"] = round(s, 4)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + (["clip"] if "clip" not in rows[0] and any("clip" in r for r in rows) else []))
        w.writeheader()
        w.writerows(rows)
    print(f"  {csv_name}: {len(sc)}枚に CLIP を付与 "
          f"(mean={np.mean(sc):.4f} min={np.min(sc):.4f} max={np.max(sc):.4f})")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["exp1_density", "exp2_scale"]):
        annotate(name)
