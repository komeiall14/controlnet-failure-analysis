"""加算元を 0 にする修正が実用になるかを確かめる。

第4.1節では、加算元を torch.zeros にすると NaN が消えることを 4 ステップで示した。
しかしそれは「壊れなくなった」までで、**実際の生成に使えるか**は見ていない。
enable_attention_slicing() はメモリを節約するための設定なので、修正して使える
ようになったなら、その節約が実際に得られていなければ改善とは呼べない。

そこで本番と同じ 20 ステップ・512×512 で、次の 3 条件を同一シード・同一
プロンプト・同一条件地図で比べる。

  (a) slicing 無効                      … 本レポートが採った回避策
  (b) slicing 有効（本体のまま）        … 公式推奨どおり。壊れるはず
  (c) slicing 有効＋加算元を 0 で初期化 … 提案する修正

見るのは 2 点。
  1. 生成が成立するか（NaN・単色でないか）
  2. 出力の質が (a) と揃うか（構造整合 edge F1 と、両者の画素差）
  ※ メモリは測っていない。torch.mps.driver_allocated_memory() はプロセス全体の
     確保量（モデルの常駐分）を返すので、生成中の活性化のピークは取れない。
     3条件とも同じ値になるだけで、slicing の効果の有無を判定できない。

出力: results/exp12_fix_usable.csv
    python3 exp12_fix_usable.py
"""
import csv
import gc
import os

import cv2
import numpy as np
import torch
from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.models.attention_processor import Attention

import dataset as ds
import pipeline as P

STEPS = 20
SEED = 0
N_IMAGES = 5


def zeros_patch(self, query, key, attention_mask=None):
    """本体と同じ手順。加算元を 0 で初期化する点だけが違う。"""
    buf = torch.zeros(query.shape[0], query.shape[1], key.shape[1],
                      dtype=query.dtype, device=query.device)
    sc = torch.baddbmm(buf, query, key.transpose(-1, -2),
                       beta=0, alpha=self.scale)
    del buf
    out = sc.softmax(dim=-1).to(query.dtype)
    del sc
    return out


def build(slicing):
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=torch.float16)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    p.scheduler = DPMSolverMultistepScheduler.from_config(
        p.scheduler.config, use_karras_sigmas=True)
    p = p.to("mps")
    p.set_progress_bar_config(disable=True)
    if slicing:
        p.enable_attention_slicing(1)
    return p


def run(label, slicing, patch, targets):
    p = build(slicing)
    orig = Attention.get_attention_scores
    if patch:
        Attention.get_attention_scores = zeros_patch
    rows = []
    try:
        for name, img, prompt, cond in targets:
            torch.mps.empty_cache()
            base = torch.mps.driver_allocated_memory()
            g = torch.Generator(device="cpu").manual_seed(SEED)
            out = p(prompt, image=P.to_cond(cond), num_inference_steps=STEPS,
                    generator=g).images[0]
            peak = torch.mps.driver_allocated_memory()
            a = np.array(out)[:, :, ::-1]                 # PIL(RGB) -> BGR
            bad = bool(np.isnan(a).any()) or float(a.std()) < 1.0
            rows.append(dict(cond=label, source=name,
                             broken=bad,
                             f1=round(P.edge_f1(cond, a)["f1"], 4),
                             mean=round(float(a.mean()), 2),
                             std=round(float(a.std()), 2),
                             mem_gb=round(peak / 1024 ** 3, 3)))
            cv2.imwrite(os.path.join(P.OUT, "images",
                                     f"e12_{label}_{name}.png"), a)
            print(f"  {label:12s} {name:12s} 壊れ={bad}  f1={rows[-1]['f1']:.4f}  "
                  f"std={rows[-1]['std']:.1f}  メモリ {rows[-1]['mem_gb']:.2f}GB",
                  flush=True)
    finally:
        Attention.get_attention_scores = orig
        del p
        gc.collect()
        torch.mps.empty_cache()
    return rows


def main():
    targets = []
    for name, img, prompt in ds.real_images()[:N_IMAGES]:
        targets.append((name, img, prompt, P.canny(img)))

    rows = []
    rows += run("slicingなし", False, False, targets)
    rows += run("slicingあり", True, False, targets)
    rows += run("slicingあり修正", True, True, targets)

    path = os.path.join(P.OUT, "exp12_fix_usable.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {path} ({len(rows)} rows)")

    import pandas as pd
    d = pd.DataFrame(rows)
    print(d.groupby("cond").agg(壊れた枚数=("broken", "sum"), n=("broken", "size"),
                                edgeF1平均=("f1", "mean"),
                                メモリGB=("mem_gb", "max")).round(4).to_string())
    pixel_diff([r["source"] for r in rows if r["cond"] == "slicingあり修正"
                and not r["broken"]])


def pixel_diff(sources=None):
    """修正版と slicing なしの画素差を CSV に残す。

    本文（第4.1節）の「画素差も平均 0.18」の根拠がこれである。
    保存した画像から計算し直せるので、生成をやり直さなくても検証できる。
    """
    import pandas as pd
    d = os.path.join(P.OUT, "images")
    if sources is None:
        sources = sorted(f[len("e12_slicingなし_"):-4] for f in os.listdir(d)
                         if f.startswith("e12_slicingなし_"))
    rows = []
    for src in sources:
        a = cv2.imread(os.path.join(d, f"e12_slicingなし_{src}.png"))
        b = cv2.imread(os.path.join(d, f"e12_slicingあり修正_{src}.png"))
        if a is None or b is None:
            continue
        g = np.abs(a.astype(float) - b.astype(float))
        rows.append(dict(source=src, mean_abs_diff=round(float(g.mean()), 4),
                         max_abs_diff=round(float(g.max()), 4)))
    if not rows:
        return print("  画素差: 比較できる画像が無い")
    t = pd.DataFrame(rows)
    f = os.path.join(P.OUT, "exp12_pixel_diff.csv")
    t.to_csv(f, index=False)
    print(f"\n-> {f} ({len(t)} rows)")
    print(f"  修正版と slicing なしの画素差（0-255）"
          f"平均 {t['mean_abs_diff'].mean():.2f}  最大 {t['max_abs_diff'].max():.0f}")


if __name__ == "__main__":
    main()
