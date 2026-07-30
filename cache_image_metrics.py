"""生成画像から導く測定値を CSV に書き出す。

results/images/ は 514 枚・158MB あるためリポジトリに含めていない。
一方 stats_report.py は第4.2節の「本来指定したかった輪郭との一致」を
生成画像から計算するので、そのままだと clone した人は再現できない。
画像に由来する値だけを表にして出しておけば、GPU も画像も無い環境で
本文の全数値を検証できる。

書き出すのは
  - e1 の各タグについて、生成画像と「各入力の輪郭」との edge F1（総当たり）
  - 生成画像の md5（密度0の重複を数えるため）
  - 生成画像のエッジ画素率

画像がある環境では stats_report.py は画像から直接計算し、この表は使わない。
どちらで計算したかは実行時に表示される。

    python3 cache_image_metrics.py
"""
import hashlib
import os

import cv2
import pandas as pd

import dataset as ds
import pipeline as P

R = P.OUT
IMG = os.path.join(R, "images")


def main():
    tbl = {n: im for n, im, _ in ds.real_images() + ds.synthetic_images()}
    refs = {n: P.canny(im) for n, im in tbl.items()}
    d = pd.read_csv(os.path.join(R, "exp1_density.csv"))
    # 第4.2節が使うのは密度0の行と α=1.0 の行
    need = d[(d["cond_density"] == 0) | (d["alpha"] == 1.0)]

    rows = []
    for r in need.itertuples():
        f = os.path.join(IMG, f"{r.tag}_gen.png")
        g = cv2.imread(f)
        if g is None:
            print(f"  画像が無い: {r.tag}")
            continue
        md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
        ge = P.edge_density(P.canny(g))
        for ref_source, ref in refs.items():
            rows.append(dict(tag=r.tag, source=r.source, alpha=r.alpha,
                             cond_density=r.cond_density, ref_source=ref_source,
                             f1=P.edge_f1(ref, g)["f1"], md5=md5,
                             gen_edge_density=ge))
    c = pd.DataFrame(rows)
    path = os.path.join(R, "image_metrics.csv")
    c.to_csv(path, index=False)
    print(f"-> {path}  {c['tag'].nunique()} タグ × {len(refs)} 参照 = {len(c)} 行")


if __name__ == "__main__":
    main()
