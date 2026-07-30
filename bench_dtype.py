"""fp16 と fp32 の生成時間を、ウォームアップつきで測る。

exp5 は NaN の発生条件を地図化するのが目的で、各条件を1回ずつしか回していない。
その時間を速度比に使ったところ、fp32 は slicing なし 45.95 秒に対し max 26.12 秒と、
分割した方が速いという並びになった。最初に回った条件がモデルのロードと MPS の
初回コンパイルを被っているためで、速度の比較には使えない。

ここでは各条件でウォームアップを1回捨ててから3回測り、中央値を採る。
slicing は使わない（fp16 では NaN になるため比較にならない）。

出力: results/bench_dtype.csv
    python3 bench_dtype.py
"""
import csv
import gc
import os
import statistics
import time

import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline)

import dataset as ds
import pipeline as P

STEPS = 4      # exp5 と同じ条件（DDIM 4ステップ）にそろえる
WARMUP = 1
REPEAT = 3


def bench(dtype_name, dtype, cond, prompt):
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=dtype)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    p = p.to("mps")
    p.set_progress_bar_config(disable=True)

    secs = []
    for i in range(WARMUP + REPEAT):
        g = torch.Generator(device="cpu").manual_seed(0)
        t0 = time.time()
        p(prompt, image=cond, num_inference_steps=STEPS, generator=g,
          output_type="latent")
        torch.mps.synchronize()
        dt = time.time() - t0
        tag = "warmup" if i < WARMUP else f"run{i - WARMUP + 1}"
        print(f"  {dtype_name} {tag}: {dt:.2f}秒", flush=True)
        if i >= WARMUP:
            secs.append(dt)
    del p, cn
    gc.collect()
    torch.mps.empty_cache()
    return secs


def main():
    name, img, prompt = ds.real_images()[0]
    cond = P.to_cond(P.canny(img))
    rows = []
    res = {}
    for dtype_name, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
        secs = bench(dtype_name, dtype, cond, prompt)
        res[dtype_name] = statistics.median(secs)
        for i, s in enumerate(secs, 1):
            rows.append(dict(dtype=dtype_name, run=i, sec=round(s, 2),
                             steps=STEPS, slicing="なし"))
    ratio = res["fp32"] / res["fp16"]
    print(f"\n  中央値 fp16={res['fp16']:.1f}秒 / fp32={res['fp32']:.1f}秒 "
          f"→ {ratio:.2f}倍")

    path = os.path.join(P.OUT, "bench_dtype.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
