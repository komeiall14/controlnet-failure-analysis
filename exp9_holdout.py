"""開発に使っていない8画像で、第6節の改善をそのまま試す。

第6節の改善と、その適用条件（適応後の密度が目標を下回るときだけ適用する）は
real_images() の10点と合成5点から作った。同じデータで評価すれば性能の
見積もりは楽観側に寄る。そこで開発に一切使っていない skimage.data の8点を
別に取り、規則を一切調整せずに当てて再現するかを見る。

手順は第6節（exp3）と完全に同じ。同一画像・同一プロンプト・同一シードで
改善前と改善後を対にし、コントラスト2水準 × シード3個を回す。
評価は共通参照（本来指定したかった輪郭）でも取る。

出力: results/exp9_holdout.csv
    python3 exp9_holdout.py
"""
import csv
import os

import cv2
import numpy as np

import dataset as ds
import exp as E
import pipeline as P

SEEDS = (0, 1, 2)
ALPHAS = (0.3, 1.0)


def main():
    # 既に測った画像は飛ばして追記する（同じ手順・同じシードなので混ぜてよい）
    path = os.path.join(P.OUT, "exp9_holdout.csv")
    rows, done = [], set()
    if os.path.exists(path):
        import csv as _csv
        with open(path, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r["source"] == "cat":      # chelsea のエイリアスなので捨てる
                    continue
                for k in ("alpha", "seed", "dens_before", "dens_after", "scale_after",
                          "f1_before", "f1_after", "f1c_before", "f1c_after"):
                    r[k] = float(r[k])
                r["seed"] = int(r["seed"])
                rows.append(r)
                done.add(r["source"])
        print(f"  既に測った {len(done)} 画像は飛ばす")
    imgs = [x for x in ds.holdout_images() if x[0] not in done]
    print(f"  検証用 {len(imgs)} 画像 × {len(ALPHAS)} 水準 × {len(SEEDS)} シード "
          f"× 2 = {len(imgs) * len(ALPHAS) * len(SEEDS) * 2} 枚")
    for name, img, prompt in imgs:
        ref = P.canny(img)                      # 共通参照＝本来指定したかった輪郭
        for a in ALPHAS:
            dim = cv2.convertScaleAbs(img, alpha=a, beta=int(128 * (1 - a)))
            e_base = P.canny(dim)
            d_base = P.edge_density(e_base)
            e_ad, d_ad = E.adaptive_canny(dim)
            sc = E.density_aware_scale(d_ad)
            for seed in SEEDS:
                gb, _ = P.generate(prompt, P.to_cond(e_base), seed, 1.0, E.STEPS)
                ga, _ = P.generate(prompt, P.to_cond(e_ad), seed, sc, E.STEPS)
                rows.append(dict(
                    source=name, alpha=a, seed=seed,
                    dens_before=d_base, dens_after=d_ad, scale_after=sc,
                    f1_before=P.edge_f1(e_base, gb)["f1"],
                    f1_after=P.edge_f1(e_ad, ga)["f1"],
                    f1c_before=P.edge_f1(ref, gb)["f1"],
                    f1c_after=P.edge_f1(ref, ga)["f1"]))
                r = rows[-1]
                print(f"  {name} a={a} s{seed}: 共通参照 "
                      f"{r['f1c_before']:.3f} -> {r['f1c_after']:.3f} "
                      f"(scale {sc:.2f}, 密度 {d_base:.4f}->{d_ad:.4f})", flush=True)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    print(f"-> results/exp9_holdout.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
