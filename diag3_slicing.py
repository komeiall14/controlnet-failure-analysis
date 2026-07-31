"""第4.1節の NaN が計算のどこで生まれるかを特定する。

入出力を観察して言えるのは「fp16 かつ attention slicing のときだけ壊れる」
「分割の粒度に依存しない」までである。なぜ壊れるのかはそこからは決まらない。

slicing を有効にすると attention の実装が SDPA から
Attention.get_attention_scores（torch.baddbmm でスコア行列を明示的に構成する経路）
へ入れ替わる。そこで get_attention_scores を包んで、呼び出しごとに

  - 入力 query / key の絶対値の最大
  - スコア行列（softmax 前）の絶対値の最大
  - 出力に NaN が入ったか

を記録し、最初に NaN を出した呼び出しの前後を調べる。fp16 の表現範囲は
±65504 なので、スコア行列がこれを超えていれば溢れが原因だと言える。
超えていなければ別の原因ということになる。

    python3 diag3_slicing.py
"""
import os

import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.models.attention_processor import Attention

import dataset as ds
import pipeline as P

FP16_MAX = 65504.0


def build(dtype, slicing):
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=dtype)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    p = p.to("mps")
    p.set_progress_bar_config(disable=True)
    if slicing:
        p.enable_attention_slicing(slicing)
    return p


def run(dtype, slicing, label):
    log = []
    orig = Attention.get_attention_scores

    def wrapped(self, query, key, attention_mask=None):
        qmax = float(query.abs().max())
        kmax = float(key.abs().max())
        # 本体と同じ式でスコア行列を作り、softmax 前の大きさを見る
        with torch.no_grad():
            sc = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1],
                            dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=self.scale)
        smax = float(sc.abs()[torch.isfinite(sc.abs())].max()) if torch.isfinite(sc).any() else float("inf")
        s_nan = bool(torch.isnan(sc).any())
        s_inf = bool(torch.isinf(sc).any())
        out = orig(self, query, key, attention_mask)
        o_nan = bool(torch.isnan(out).any())
        log.append(dict(i=len(log), q=qmax, k=kmax, score_max=smax,
                        score_nan=s_nan, score_inf=s_inf, out_nan=o_nan,
                        shape=tuple(query.shape), scale=self.scale))
        return out

    Attention.get_attention_scores = wrapped
    try:
        name, img, prompt = ds.real_images()[0]
        cond = P.to_cond(P.canny(img))
        p = build(torch.float16 if dtype == "fp16" else torch.float32, slicing)
        g = torch.Generator(device="cpu").manual_seed(0)
        lat = p(prompt, image=cond, num_inference_steps=1, generator=g,
                output_type="latent").images
        latent_nan = bool(torch.isnan(lat).any())
    finally:
        Attention.get_attention_scores = orig

    print(f"\n===== {label} =====")
    if not log:
        print("  get_attention_scores は呼ばれなかった（SDPA 経路）")
        print(f"  潜在変数に NaN: {latent_nan}")
        return log

    print(f"  get_attention_scores の呼び出し {len(log)} 回")
    print(f"  潜在変数に NaN: {latent_nan}")
    fin = [r for r in log if r["score_max"] != float("inf")]
    print(f"  スコア行列の絶対値の最大: {max(r['score_max'] for r in fin):.1f}"
          f"（fp16 の上限 {FP16_MAX:.0f}）")
    print(f"  query の最大: {max(r['q'] for r in log):.1f} / "
          f"key の最大: {max(r['k'] for r in log):.1f}")
    n_inf = sum(r["score_inf"] for r in log)
    n_nan = sum(r["score_nan"] for r in log)
    n_out = sum(r["out_nan"] for r in log)
    print(f"  スコア行列が inf を含む呼び出し: {n_inf}/{len(log)}")
    print(f"  スコア行列が NaN を含む呼び出し: {n_nan}/{len(log)}")
    print(f"  出力（softmax 後）が NaN の呼び出し: {n_out}/{len(log)}")

    first = next((r for r in log if r["out_nan"]), None)
    if first:
        print(f"\n  最初に出力が NaN になった呼び出し #{first['i']}")
        print(f"    query shape={first['shape']}  scale={first['scale']:.5f}")
        print(f"    |query|max={first['q']:.1f}  |key|max={first['k']:.1f}")
        print(f"    スコア行列 max={first['score_max']:.1f}  "
              f"inf={first['score_inf']}  nan={first['score_nan']}")
        prev = [r for r in log if r["i"] < first["i"]]
        if prev:
            print(f"    それ以前 {len(prev)} 回のスコア最大: "
                  f"{max(r['score_max'] for r in prev if r['score_max'] != float('inf')):.1f}")
    return log


def main():
    run("fp16", False, "fp16 / slicing なし（既定の SDPA 経路）")
    run("fp32", 1, "fp32 / slicing あり（正常に動く側）")
    run("fp16", 1, "fp16 / slicing あり（壊れる側）")


if __name__ == "__main__":
    main()
