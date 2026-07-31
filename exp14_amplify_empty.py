"""条件地図が空になった入力で scale を上げても指定が戻らないかを確かめる。

第8節は「密度が低い入力でも scale を上げれば構造整合は改善するが、条件地図が
空になった入力では増幅しても戻らない」と述べていた。前半は掃引で測っているが、
後半は第5節（残差の条件由来の成分が縮む）からの推論だった。空の条件地図を
与えたまま scale を振れば直接測れる。

条件地図が空になる入力（α=0.15 で密度 0）から実画像 2 枚を選び、
scale を 4 段階に振って、本来指定したかった輪郭（コントラストを落とさない
条件地図）との edge F1 を測る。比較のため、同じ入力の α=1.0（条件地図あり）も
同じ scale で振る。

出力: results/exp14_amplify_empty.csv
    python3 exp14_amplify_empty.py
"""
import csv
import os

import cv2

import dataset as ds
import pipeline as P

SCALES = (0.2, 1.0, 2.0, 3.0)
SEED = 0
SOURCES = ("camera", "rocket")


def main():
    os.makedirs(os.path.join(P.OUT, "images"), exist_ok=True)
    tbl = {n: (im, pr) for n, im, pr in ds.real_images()}
    rows = []
    for src in SOURCES:
        img, prompt = tbl[src]
        ref = P.canny(img)                       # 本来指定したかった輪郭
        for alpha in (0.15, 1.0):
            a, dark = ds.contrast_series(img, alphas=(alpha,))[0]
            e = P.canny(dark)
            dens = P.edge_density(e)
            for sc in SCALES:
                gen, _ = P.generate(prompt, P.to_cond(e), SEED, scale=sc,
                                    steps=20)
                f1 = P.edge_f1(ref, gen)["f1"]   # 参照は常に本来の輪郭
                rows.append(dict(source=src, alpha=alpha, cond_density=round(dens, 6),
                                 scale=sc, f1_vs_intended=round(f1, 4)))
                print(f"  {src:8s} α={alpha:<5} 密度={dens:.5f} scale={sc:<4} "
                      f"本来の輪郭との F1={f1:.4f}")
                cv2.imwrite(os.path.join(P.OUT, "images",
                                         f"e14_{src}_a{alpha}_s{sc}.png"), gen)

    f = os.path.join(P.OUT, "exp14_amplify_empty.csv")
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {f}")
    for src in SOURCES:
        for alpha in (0.15, 1.0):
            v = [r["f1_vs_intended"] for r in rows
                 if r["source"] == src and r["alpha"] == alpha]
            print(f"  {src:8s} α={alpha:<5} scale {SCALES[0]}→{SCALES[-1]} で "
                  f"{v[0]:.4f}→{v[-1]:.4f}（幅 {max(v) - min(v):.4f}）")


if __name__ == "__main__":
    main()
