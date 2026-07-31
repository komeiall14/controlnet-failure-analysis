"""attention slicing が実際にメモリを節約するかを測る。

第4.1節の修正は「推奨されている省メモリ設定を使えるようにする」ものなので、
使えるようになった状態で節約が得られているかを確かめる。

torch.mps.driver_allocated_memory() はプロセス全体の確保量（モデルの常駐分を
含む）を返すため、条件による差が埋もれる。torch.mps.current_allocated_memory()
はテンソルが確保している量を返し、確保・解放に追随するので、生成中に別スレッド
から刻んで最大値を取る。torch 2.2.2 の MPS にはピークを直接返す API が無い。

標本化は山を取り逃すことしかできないので、観測された最大値は真の山の下界である。
刻みを 5ms にすると slicing 側が 0.44〜0.66 GiB の間で毎回変わり、取り逃しが
大きい。0.2ms まで細かくすると 0.907 GiB に何度も一致して収束するので、そちらを
採る。さらに条件ごとに REPEAT 回まわして最大値を採る。

比べるのは3条件。
  (a) slicing 無効                      … 本レポートが採った回避策
  (b) slicing 有効（本体のまま）        … 公式推奨どおり。壊れる
  (c) slicing 有効＋加算元を 0 で初期化 … 提案する修正

出力: results/exp13_memory.csv
    python3 exp13_memory.py
"""
import csv
import os
import threading
import time

import numpy as np
import torch
from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.models.attention_processor import Attention

import dataset as ds
import pipeline as P

STEPS = 8          # 山の高さは1ステップで決まるので短くてよい
REPEAT = 3         # 標本化は取り逃すだけなので、複数回の最大を採る
PROMPT = "a photograph of a man with a camera on a tripod"


def patched_get_attention_scores(self, query, key, attention_mask=None):
    """加算元を 0 で初期化した get_attention_scores（第4.1節の修正）。"""
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()
    baddbmm_input = torch.zeros(query.shape[0], query.shape[1], key.shape[1],
                                dtype=query.dtype, device=query.device)
    beta = 0
    attention_scores = torch.baddbmm(
        baddbmm_input, query, key.transpose(-1, -2), beta=beta, alpha=self.scale)
    del baddbmm_input
    if self.upcast_softmax:
        attention_scores = attention_scores.float()
    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores
    return attention_probs.to(dtype)


class Peak:
    """生成中の確保量を刻んで最大値を取る。"""

    def __init__(self, dt=0.0002):
        self.dt, self.on, self.peak = dt, False, 0

    def __enter__(self):
        torch.mps.empty_cache()
        self.base = torch.mps.current_allocated_memory()
        self.peak = self.base
        self.on = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self):
        while self.on:
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(self.dt)

    def __exit__(self, *a):
        self.on = False
        self.t.join()

    @property
    def gib(self):
        return (self.peak - self.base) / 1024 ** 3


def build():
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=torch.float16)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    p.scheduler = DPMSolverMultistepScheduler.from_config(
        p.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    p = p.to("mps")
    p.set_progress_bar_config(disable=True)
    return p


def main():
    img = {n: im for n, im, _ in ds.real_images()}["camera"]
    cond = P.to_cond(P.canny(img))
    p = build()
    orig = Attention.get_attention_scores
    rows = []

    for name, slicing, patch in (("slicingなし", False, False),
                                 ("slicingあり", True, False),
                                 ("slicingあり修正", True, True)):
        if slicing:
            p.enable_attention_slicing()
        else:
            p.disable_attention_slicing()
        Attention.get_attention_scores = (patched_get_attention_scores if patch
                                          else orig)
        peaks, broken, std = [], None, None
        for _ in range(REPEAT):
            g = torch.Generator(device="cpu").manual_seed(0)
            with Peak() as pk:
                out = p(PROMPT, image=cond, num_inference_steps=STEPS,
                        generator=g).images[0]
            peaks.append(pk.gib)
            rgb = np.array(out)
            broken = (not np.isfinite(rgb).all()) or float(rgb.std()) < 1.0
            std = round(float(rgb.std()), 2)
            torch.mps.empty_cache()
        rows.append(dict(cond=name, peak_gib=round(max(peaks), 4),
                         runs=REPEAT, spread=round(max(peaks) - min(peaks), 4),
                         broken=broken, std=std))
        print(f"  {name:16s} ピーク {max(peaks):.3f} GiB"
              f"（{REPEAT}回の最大。振れ幅 {max(peaks) - min(peaks):.3f}）"
              f"  壊れた={broken}")
        Attention.get_attention_scores = orig
        torch.mps.empty_cache()

    f = os.path.join(P.OUT, "exp13_memory.csv")
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {f}")
    a = {r["cond"]: r["peak_gib"] for r in rows}
    if "slicingなし" in a and "slicingあり修正" in a and a["slicingなし"] > 0:
        d = a["slicingなし"] - a["slicingあり修正"]
        print(f"  修正版は slicing 無効時より {d:+.3f} GiB"
              f"（{d / a['slicingなし'] * 100:+.1f}%）")


if __name__ == "__main__":
    main()
