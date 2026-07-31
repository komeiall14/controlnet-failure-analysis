"""第6節の改善を、二つの弱点を埋めて評価し直す。

弱点1: 評価が構造整合（edge F1）だけで、プロンプト追従を測っていない。
    第8節は「第4.3節の『上げるほど良い』は片方の指標しか見ていなかったから」と
    結論しているのに、改善策の評価が同じ誤りを繰り返している。
    生成済み画像は results/images に残っているので、拡散を回さずに CLIP を測れる。

弱点2: 本文が最終的に推奨した形（密度が目標を下回るときにだけ適用する）を
    評価していない。無条件適用の数字しか出していない。手元のデータで計算できる。

あわせて、前後を共通の参照（本来指定したかった輪郭）で測った値も出す。
それぞれが与えられた地図で測ると、F1 の密度依存が改善幅に混入するためである。

出力: results/exp7_policy.csv
    python3 exp7_policy.py
"""
import csv
import os

import cv2
import numpy as np
import pandas as pd
from scipy import stats

import clip_score as CS
import dataset as ds
import pipeline as P

R = P.OUT
IMG = os.path.join(R, "images")
TARGET = 0.045          # adaptive_canny / density_aware_scale と同じ目標


def main():
    tbl = {n: (im, pr) for n, im, pr in ds.real_images() + ds.synthetic_images()}
    d = pd.read_csv(os.path.join(R, "exp3_improvement.csv"))

    paths, prompts, keys = [], [], []
    rows = []
    for r in d.itertuples():
        img, prompt = tbl[r.source]
        ref = P.canny(img)                    # 共通参照＝本来指定したかった輪郭
        fb = os.path.join(IMG, f"e3_{r.source}_a{r.alpha}_s{r.seed}_before.png")
        fa = os.path.join(IMG, f"e3_{r.source}_a{r.alpha}_s{r.seed}_after.png")
        gb, ga = cv2.imread(fb), cv2.imread(fa)
        if gb is None or ga is None:
            continue
        rows.append(dict(source=r.source, alpha=r.alpha, seed=r.seed,
                         dens_before=r.dens_before, dens_after=r.dens_after,
                         scale_after=r.scale_after,
                         f1_before=r.f1_before, f1_after=r.f1_after,
                         f1c_before=P.edge_f1(ref, gb)["f1"],
                         f1c_after=P.edge_f1(ref, ga)["f1"]))
        for f, side in ((fb, "before"), (fa, "after")):
            paths.append(f)
            prompts.append(prompt)
            keys.append((len(rows) - 1, side))

    print(f"  CLIP を測る画像: {len(paths)} 枚")
    scores = CS.score(paths, prompts)
    for (i, side), sc in zip(keys, scores):
        rows[i][f"clip_{side}"] = float(sc)

    c = pd.DataFrame(rows)
    path = os.path.join(R, "exp7_policy.csv")
    c.to_csv(path, index=False)
    print(f"-> {path} ({len(c)} rows)\n")

    # ---- 1. プロンプト追従を含めた前後比較 -------------------------------
    print("【改善前後：構造整合とプロンプト追従の両方】（画像単位・α層別）")
    for a, g in c.groupby("alpha"):
        p = g.groupby("source").agg(
            f1b=("f1_before", "mean"), f1a=("f1_after", "mean"),
            cb=("clip_before", "mean"), ca=("clip_after", "mean"))
        wf = stats.wilcoxon(p["f1a"], p["f1b"])
        wc = stats.wilcoxon(p["ca"], p["cb"])
        print(f"  α={a}  n={len(p)}画像")
        print(f"    構造整合   {p['f1b'].mean():.4f} → {p['f1a'].mean():.4f}  "
              f"Δ中央値={(p['f1a'] - p['f1b']).median():+.4f}  p={wf.pvalue:.4f}")
        print(f"    CLIP       {p['cb'].mean():.4f} → {p['ca'].mean():.4f}  "
              f"Δ中央値={(p['ca'] - p['cb']).median():+.4f}  p={wc.pvalue:.4f}  "
              f"（悪化した画像 {int(((p['ca'] - p['cb']) < 0).sum())}/{len(p)}）")

    # ---- 2. 推奨形（密度が目標を下回るときだけ適用）---------------------
    print("\n【推奨形：改善前の密度が目標 0.045 を下回るときだけ適用】")
    print("  ※ 適用しない画像は改善前の値をそのまま採る")
    for a, g in c.groupby("alpha"):
        p = g.groupby("source").agg(
            db=("dens_before", "first"),
            f1b=("f1_before", "mean"), f1a=("f1_after", "mean"),
            fcb=("f1c_before", "mean"), fca=("f1c_after", "mean"),
            cb=("clip_before", "mean"), ca=("clip_after", "mean"))
        use = p["db"] < TARGET                      # 適用する画像
        pol_f1 = np.where(use, p["f1a"], p["f1b"])
        pol_fc = np.where(use, p["fca"], p["fcb"])
        pol_cl = np.where(use, p["ca"], p["cb"])
        wf = stats.wilcoxon(pol_f1, p["f1b"])
        wc = stats.wilcoxon(pol_fc, p["fcb"])
        wl = stats.wilcoxon(pol_cl, p["cb"])
        print(f"  α={a}  適用 {int(use.sum())}/{len(p)}画像")
        print(f"    構造整合(各自の地図)  Δ中央値={np.median(pol_f1 - p['f1b']):+.4f}"
              f"  p={wf.pvalue:.4f}")
        print(f"    構造整合(共通参照)    Δ中央値={np.median(pol_fc - p['fcb']):+.4f}"
              f"  p={wc.pvalue:.4f}")
        print(f"    CLIP                  Δ中央値={np.median(pol_cl - p['cb']):+.4f}"
              f"  p={wl.pvalue:.4f}")


if __name__ == "__main__":
    main()
