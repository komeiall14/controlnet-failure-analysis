"""残差の空間構造を測る。

第5節で「条件地図が空でも残差は一様ではない」と述べるための測定。
probe.py は残差の大きさ（RMS）しか出さないので、その残差が空間的に
変化しているのか、チャネルごとに一様な場なのかは分からない。

各段の残差 r[b, c, h, w] について、チャネルごとの空間平均を引いた

    r - mean_{h,w}(r)

の RMS を測り、元の RMS に対する比を「空間成分の割合」とする。
比が 0 に近ければチャネルごとに一様な場、1 に近ければ空間的に変化している。

測定条件は probe.py と同一（fp16 / MPS / t=500 / seed0 の潜在変数）。

出力: results/probe_spatial.csv
    python3 probe_spatial.py
"""
import csv
import os

import cv2
import numpy as np
import torch

import dataset as ds
import pipeline as P
import probe


@torch.no_grad()
def spatial_split(p, edges, prompt, t=500, seed=0):
    """残差を「チャネルごとに一様な成分」と「空間的に変化する成分」に分ける。"""
    cond = torch.from_numpy(
        np.stack([edges] * 3, 0)[None].astype(np.float32) / 255.0
    ).to("mps", torch.float16)
    emb, _ = p.encode_prompt(prompt, "mps", 1, False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(1, 4, 64, 64, generator=g).to("mps", torch.float16)
    ts = torch.tensor([t], device="mps")
    down, mid = p.controlnet(lat, ts, encoder_hidden_states=emb,
                             controlnet_cond=cond, return_dict=False)

    def split(x):
        x = x.float()
        rms = float(x.pow(2).mean().sqrt())
        # チャネルごとの空間平均を引く＝空間的に変化している分だけを残す
        var = x - x.mean(dim=(-2, -1), keepdim=True)
        vrms = float(var.pow(2).mean().sqrt())
        return rms, vrms, (vrms / rms if rms > 0 else float("nan"))

    out = {}
    for i, d in enumerate(down):
        r, v, f = split(d)
        out[f"down{i}_rms"] = r
        out[f"down{i}_spatial_rms"] = v
        out[f"down{i}_spatial_frac"] = f
    r, v, f = split(mid)
    out.update(mid_rms=r, mid_spatial_rms=v, mid_spatial_frac=f)
    return out


def main():
    p = probe.build()
    tbl = {n: (im, pr) for n, im, pr in ds.real_images() + ds.synthetic_images()}
    rows = []

    # (1) 完全に空の条件地図。プロンプトは probe.py の参照と同じ "a photo"
    empty = np.zeros((512, 512), np.uint8)
    r = spatial_split(p, empty, "a photo")
    r.update(source="__empty__", prompt="a photo", density=0.0)
    rows.append(r)

    # (2) 実際の条件地図。camera を代表として原画像から Canny を取る
    for name in ("camera", "astronaut", "coins"):
        if name not in tbl:
            continue
        img, prompt = tbl[name]
        e = P.canny(img)
        r = spatial_split(p, e, prompt)
        r.update(source=name, prompt=prompt, density=P.edge_density(e))
        rows.append(r)

    # (3) コントラストを落として密度0になった camera（本文 §4.2 の破綻条件）
    img, prompt = tbl["camera"]
    dim = cv2.convertScaleAbs(img, alpha=0.15, beta=int(128 * (1 - 0.15)))
    e = P.canny(dim)
    r = spatial_split(p, e, prompt)
    r.update(source="camera_a0.15", prompt=prompt, density=P.edge_density(e))
    rows.append(r)

    for x in rows:
        print(f"  {x['source']:16s} dens={x['density']:.5f}  "
              f"mid RMS={x['mid_rms']:.4f} 空間成分={x['mid_spatial_rms']:.4f} "
              f"({x['mid_spatial_frac'] * 100:.1f}%)  "
              f"down6 {x['down6_spatial_frac'] * 100:.1f}%", flush=True)

    path = os.path.join(P.OUT, "probe_spatial.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
