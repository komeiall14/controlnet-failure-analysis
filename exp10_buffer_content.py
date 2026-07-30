"""加算元バッファの中身を直接見る。

第4.1節は「加算元を差し替えると NaN が消える」ところまでを測ったが、
バッファの中身そのものは観測していなかった。そのため
「非有限のビット列が残っているから汚染される」は仮説にとどまっていた。

ここでは get_attention_scores を、本体と同じ手順のまま
（torch.empty で確保 → baddbmm → softmax、直後に del）差し替え、
baddbmm へ渡す直前のバッファに isfinite をかけて非有限要素を数える。
本体との差は計測を挟む点だけで、確保のされ方も寿命も変えない。

fp16 と fp32 の両方で測る。
  - fp16 でだけ非有限が出る → 「非有限のビット列」説が確定する
  - どちらにも出ない → 説は誤りで、別の機構（beta=0 でも加算元を読む等）になる

出力: results/exp10_buffer_content.csv
    python3 exp10_buffer_content.py
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

STEPS = 4


def run(dtype_name):
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=dtype)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    p = p.to("mps")
    p.set_progress_bar_config(disable=True)
    p.enable_attention_slicing(1)

    orig = Attention.get_attention_scores
    st = {"calls": 0, "dirty": 0, "elems": 0, "bad": 0,
          "first_dirty": None, "first_nan_out": None, "absmax": 0.0}

    def probe(self, query, key, attention_mask=None):
        # 本体（attention_processor.py L566-582）と同じ手順。
        # 違いは baddbmm の直前にバッファを検査する点だけ。
        buf = torch.empty(query.shape[0], query.shape[1], key.shape[1],
                          dtype=query.dtype, device=query.device)
        fin = torch.isfinite(buf)
        nbad = int((~fin).sum())
        st["elems"] += buf.numel()
        st["bad"] += nbad
        if nbad:
            st["dirty"] += 1
            if st["first_dirty"] is None:
                st["first_dirty"] = st["calls"]
        f = buf[fin]
        if f.numel():
            st["absmax"] = max(st["absmax"], float(f.abs().max()))
        sc = torch.baddbmm(buf, query, key.transpose(-1, -2),
                           beta=0, alpha=self.scale)
        del buf
        out = sc.softmax(dim=-1).to(query.dtype)
        del sc
        if st["first_nan_out"] is None and bool(torch.isnan(out).any()):
            st["first_nan_out"] = st["calls"]
        st["calls"] += 1
        return out

    Attention.get_attention_scores = probe
    try:
        name, img, prompt = ds.real_images()[0]
        g = torch.Generator(device="cpu").manual_seed(0)
        lat = p(prompt, image=P.to_cond(P.canny(img)), num_inference_steps=STEPS,
                generator=g, output_type="latent").images
        nan = bool(torch.isnan(lat).any())
    finally:
        Attention.get_attention_scores = orig
        del p, cn
        gc.collect()
        torch.mps.empty_cache()

    r = dict(dtype=dtype_name, calls=st["calls"], latent_nan=nan,
             buffers_with_nonfinite=st["dirty"],
             elements_checked=st["elems"], nonfinite_elements=st["bad"],
             nonfinite_ratio=round(st["bad"] / max(st["elems"], 1), 12),
             first_dirty_call=st["first_dirty"],
             first_nan_out_call=st["first_nan_out"],
             finite_absmax=round(st["absmax"], 4))
    print(f"  {dtype_name}: 呼び出し {r['calls']}  潜在変数NaN={nan}")
    print(f"    非有限を含むバッファ {r['buffers_with_nonfinite']}/{r['calls']}")
    print(f"    非有限の要素 {r['nonfinite_elements']} / {r['elements_checked']}")
    print(f"    最初に非有限が出た呼び出し {r['first_dirty_call']}  "
          f"／ 最初に出力が NaN になった呼び出し {r['first_nan_out_call']}")
    print(f"    バッファ内の有限値の絶対値最大 {r['finite_absmax']}", flush=True)
    return r


def main():
    rows = [run("fp16"), run("fp32")]
    path = os.path.join(P.OUT, "exp10_buffer_content.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")
    a, b = rows
    if a["nonfinite_elements"] and not b["nonfinite_elements"]:
        print("\n  → fp16 のバッファにだけ非有限値がある。「非有限のビット列」説を支持する")
    elif not a["nonfinite_elements"] and not b["nonfinite_elements"]:
        print("\n  → どちらのバッファにも非有限値が無い。"
              "「非有限のビット列」説は成り立たない。別の機構を考える必要がある")
    else:
        print("\n  → 両方に非有限値がある。fp16 でだけ壊れる理由は別に要る")


if __name__ == "__main__":
    main()
