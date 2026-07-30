"""第4.1節の NaN の原因を特定し、修正を検証する。

これまで分かっていたのは「fp16 かつ attention slicing のときだけ壊れる」
「分割の粒度に依存しない」までで、なぜ壊れるのかは特定できていなかった。
本文で「溢れ」と書きかけたが、下の測定はそれを否定する。

slicing を有効にすると attention の実装が SDPA から
Attention.get_attention_scores へ入れ替わる。その実装は

    baddbmm_input = torch.empty(...)        # 未初期化
    attention_scores = torch.baddbmm(baddbmm_input, query, key.T,
                                     beta=0, alpha=self.scale)

となっている。beta=0 なので数学的には加算元の中身は結果に影響しないが、
IEEE 754 では 0 × NaN = NaN、0 × Inf = NaN であり、未初期化バッファに
非有限のビット列が残っていれば出力全体が NaN になる。

測ること:
  A. NaN が最初に現れる呼び出しと、そのときの入力の大きさ
     （溢れなら入力かスコア行列が fp16 の上限 65504 に近いはず）
  B. 加算元を torch.zeros に差し替えるだけで NaN が消えるか
     （消えるなら原因は加算元の中身で確定する）

出力: results/exp8_slicing_cause.csv
    python3 exp8_slicing_cause.py
"""
import csv
import gc
import os

import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.models.attention_processor import Attention

import dataset as ds
import pipeline as P

FP16_MAX = 65504.0
STEPS = 4


def build(dtype, slicing):
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=dtype)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    p = p.to("mps")
    p.set_progress_bar_config(disable=True)
    if slicing is not None:
        p.enable_attention_slicing(slicing)
    return p


def zeros_patch(self, query, key, attention_mask=None):
    """本体と同じ計算。違いは加算元を zeros にする点だけ。"""
    buf = torch.zeros(query.shape[0], query.shape[1], key.shape[1],
                      dtype=query.dtype, device=query.device)
    sc = torch.baddbmm(buf, query, key.transpose(-1, -2),
                       beta=0, alpha=self.scale)
    return sc.softmax(dim=-1).to(query.dtype)


def trial(dtype_name, slicing, patch, seed=0):
    """1条件を回し、NaN の有無と、最初に NaN を出した呼び出しの情報を返す。"""
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    p = build(dtype, slicing)
    orig = Attention.get_attention_scores
    state = {"n": 0, "first": None}

    def probe(self, query, key, attention_mask=None):
        out = (zeros_patch if patch else orig)(self, query, key, attention_mask)
        i = state["n"]
        state["n"] += 1
        if state["first"] is None and bool(torch.isnan(out).any()):
            state["first"] = dict(
                call=i,
                q_has_nan=bool(torch.isnan(query).any()),
                k_has_nan=bool(torch.isnan(key).any()),
                q_absmax=round(float(query.float().abs().max()), 3),
                k_absmax=round(float(key.float().abs().max()), 3),
                q_len=query.shape[1], dim=query.shape[2])
        return out

    Attention.get_attention_scores = probe
    try:
        name, img, prompt = ds.real_images()[0]
        g = torch.Generator(device="cpu").manual_seed(seed)
        lat = p(prompt, image=P.to_cond(P.canny(img)), num_inference_steps=STEPS,
                generator=g, output_type="latent").images
        nan = bool(torch.isnan(lat).any())
        amax = float(lat.float().abs().max()) if not nan else float("nan")
    finally:
        Attention.get_attention_scores = orig
        del p
        gc.collect()
        torch.mps.empty_cache()

    f = state["first"] or {}
    return dict(dtype=dtype_name, slicing=str(slicing), buffer="zeros" if patch else "empty",
                seed=seed, calls=state["n"], latent_nan=nan,
                latent_absmax=None if nan else round(amax, 4),
                first_nan_call=f.get("call"), q_has_nan=f.get("q_has_nan"),
                k_has_nan=f.get("k_has_nan"), q_absmax=f.get("q_absmax"),
                k_absmax=f.get("k_absmax"))


def main():
    rows = []
    plan = [
        # 参照：壊れない組み合わせ
        ("fp16", None, False), ("fp32", 1, False),
        # 壊れる組み合わせ。粒度を変えても同じか
        ("fp16", 1, False), ("fp16", 4, False), ("fp16", "auto", False),
        # 加算元を zeros にするだけで直るか。シードを変えても再現するか
        ("fp16", 1, True), ("fp16", 4, True), ("fp16", "auto", True),
    ]
    for dtype_name, slicing, patch in plan:
        r = trial(dtype_name, slicing, patch)
        rows.append(r)
        print(f"  {r['dtype']:5s} slicing={r['slicing']:5s} buffer={r['buffer']:5s} "
              f"→ NaN={r['latent_nan']}  absmax={r['latent_absmax']}  "
              f"最初のNaN=呼び出し{r['first_nan_call']}", flush=True)

    # zeros 版がシードを変えても安定して直るか
    for seed in (1, 2):
        r = trial("fp16", 1, True, seed=seed)
        rows.append(r)
        print(f"  fp16  slicing=1     buffer=zeros seed={seed} "
              f"→ NaN={r['latent_nan']}  absmax={r['latent_absmax']}", flush=True)

    path = os.path.join(P.OUT, "exp8_slicing_cause.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)")

    bad = [r for r in rows if r["latent_nan"] and r["first_nan_call"] is not None]
    if bad:
        m = max(max(r["q_absmax"] or 0, r["k_absmax"] or 0) for r in bad)
        print(f"\n  NaN が出た条件での入力の絶対値の最大: {m}"
              f"（fp16 の上限は {FP16_MAX:.0f}）")
        print("  → 入力が fp16 の表現範囲に対して十分小さいので、溢れでは説明できない")


if __name__ == "__main__":
    main()
