"""機構の直接証拠：条件マップの密度と ControlNet 残差ノルムの関係。

ControlNet は trainable copy の各段の出力を zero-convolution に通し、
凍結された UNet の skip 接続へ残差として加算する。条件が実質的に効くかどうかは
この残差の大きさで決まるはずである。したがって

    「エッジ密度が下がると条件が無視される」

という現象は、入出力の相関ではなく **残差ノルムが密度とともに縮む** ことで
直接示せる。ControlNet 単体の1回の順伝播で測れるので、全拡散過程を回す
（1枚100秒）必要がなく、密に掃引できる。

出力: results/probe_residual.csv
"""
import csv
import os

import cv2
import numpy as np
import torch

import dataset as ds
import pipeline as P


def build():
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=torch.float16)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("mps")
    p.set_progress_bar_config(disable=True)
    return p


@torch.no_grad()
def residual_norms(p, edges, prompt, t=500, seed=0):
    """ControlNet を1回だけ順伝播させ、UNet へ注入される残差の大きさを測る。"""
    cond = torch.from_numpy(
        np.stack([edges] * 3, 0)[None].astype(np.float32) / 255.0
    ).to("mps", torch.float16)
    # _encode_prompt は非推奨のプライベート関数で、CFG無効時に None を含むタプルを
    # 連結しようとして落ちる。公開APIの encode_prompt を使い、負側は受け取らない。
    emb, _ = p.encode_prompt(prompt, "mps", 1, False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(1, 4, 64, 64, generator=g).to("mps", torch.float16)
    ts = torch.tensor([t], device="mps")
    down, mid = p.controlnet(lat, ts, encoder_hidden_states=emb,
                             controlnet_cond=cond, return_dict=False)
    # 各段の残差を二乗和で束ねる。段ごとに解像度が違うので要素数で正規化する。
    per = [float(d.float().pow(2).mean().sqrt()) for d in down]
    return dict(
        mid_rms=float(mid.float().pow(2).mean().sqrt()),
        down_rms_mean=float(np.mean(per)),
        down_rms_first=per[0], down_rms_last=per[-1],
        total_rms=float(np.sqrt(np.mean([x ** 2 for x in per + [
            float(mid.float().pow(2).mean().sqrt())]]))),
    )


def main():
    p = build()
    rows = []
    for name, img, prompt in ds.synthetic_images() + ds.real_images():
        # コントラストを細かく振って密度を連続的に動かす
        for a in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0):
            dim = cv2.convertScaleAbs(img, alpha=a, beta=int(128 * (1 - a)))
            e = P.canny(dim)
            d = P.edge_density(e)
            r = residual_norms(p, e, prompt)
            r.update(source=name, alpha=a, density=d)
            rows.append(r)
            print(f"  {name:22s} a={a:<5} dens={d:.5f} "
                  f"total_rms={r['total_rms']:.4f} mid={r['mid_rms']:.4f}",
                  flush=True)
    # 参考: 完全に空の条件マップ（密度0）での残差
    empty = np.zeros((512, 512), np.uint8)
    r = residual_norms(p, empty, "a photo")
    r.update(source="__empty__", alpha=0.0, density=0.0)
    rows.append(r)
    print(f"  {'__empty__':22s} dens=0.00000 total_rms={r['total_rms']:.4f}")

    path = os.path.join(P.OUT, "probe_residual.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
