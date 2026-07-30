"""レポート本文に載せた統計量を再現する。

analysis.py は図を描くためのもので、fig1 と fig3 には本文が採らなかった扱い
（定義上 0 になる条件を含む相関、1画像6行をプールした検定）が残っている。
本文の数値の出どころはこのスクリプトである。

    python3 stats_report.py
"""
import glob
import hashlib
import os

import cv2
import numpy as np
import pandas as pd
from scipy import stats

import dataset as ds
import pipeline as P

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(n):
    return pd.read_csv(os.path.join(R, f"{n}.csv"))


def head(t):
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


def sec3_env():
    head("第3節 生成速度（20ステップ・512×512）")
    s = []
    for f in glob.glob(os.path.join(R, "*.csv")):
        # exp5 は第4.1節の切り分け用で DDIM 4ステップ。条件が違うので混ぜない
        if os.path.basename(f).startswith("exp5_"):
            continue
        d = pd.read_csv(f)
        if "sec" in d.columns:
            s += d["sec"].dropna().tolist()
    s = np.array(s)
    print(f"  n={len(s)}  min={s.min():.1f}  max={s.max():.1f}  "
          f"中央値={np.median(s):.1f}  平均={s.mean():.1f}")


def sec42_density():
    head("第4.2節 条件地図の密度と構造整合")
    d = load("exp1_density")
    r_all = stats.spearmanr(d["cond_density"], d["f1"])
    nz = d[d["cond_density"] > 0]
    r_nz = stats.spearmanr(nz["cond_density"], nz["f1"])
    print(f"  全60条件            rho={r_all.correlation:+.3f} p={r_all.pvalue:.2g}")
    print(f"  密度>0の45条件のみ  rho={r_nz.correlation:+.3f} p={r_nz.pvalue:.2g}")
    print("  ※ 条件地図が空だと edge F1 は定義上 0 を返すので、前者は押し上げられている")

    b = pd.cut(d["cond_density"], [-1e-9, 1e-9, .005, .02, .05, 1],
               labels=["0", "〜0.005", "〜0.02", "〜0.05", "0.05〜"])
    print("\n  密度帯ごと（密度0の行は指標が使えないので参考値）")
    print(d.groupby(b, observed=True)["f1"]
          .agg(n="size", 平均="mean", 中央値="median", 標準偏差="std").round(4).to_string())

    # 条件が効いていないことの証拠：本来指定したかった輪郭との一致
    tbl = {n: im for n, im, _ in ds.real_images() + ds.synthetic_images()}

    def f1_vs_target(tag, src):
        g = cv2.imread(os.path.join(R, "images", f"{tag}_gen.png"))
        return None if g is None else P.edge_f1(P.canny(tbl[src]), g)["f1"]

    z = d[d["cond_density"] == 0]
    zs = [x for x in (f1_vs_target(r.tag, r.source) for r in z.itertuples()) if x is not None]
    ok = d[d["alpha"] == 1.0]
    os_ = [x for x in (f1_vs_target(r.tag, r.source) for r in ok.itertuples()) if x is not None]
    floor = []
    for r in ok.itertuples():
        g = cv2.imread(os.path.join(R, "images", f"{r.tag}_gen.png"))
        if g is None:
            continue
        for other in tbl:
            if other != r.source:
                floor.append(P.edge_f1(P.canny(tbl[other]), g)["f1"])
    u = stats.mannwhitneyu(zs, floor)
    print(f"\n  本来の輪郭との一致  正しく条件づけ={np.mean(os_):.4f} (n={len(os_)})")
    print(f"                      密度0        ={np.mean(zs):.4f} (n={len(zs)})")
    print(f"                      偶然の水準    ={np.mean(floor):.4f} (n={len(floor)})")
    print(f"                      Mann-Whitney p={u.pvalue:.3f}")

    # md5 重複を除いた独立な8枚
    seen = {}
    for r in z.itertuples():
        f = os.path.join(R, "images", f"{r.tag}_gen.png")
        if not os.path.exists(f):
            continue
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h not in seen:
            seen[h] = f1_vs_target(r.tag, r.source)
    v = np.array([x for x in seen.values() if x is not None])
    u8 = stats.mannwhitneyu(v, floor)
    print(f"  重複を除いた独立な生成 n={len(v)} 平均={v.mean():.4f} "
          f"中央値={np.median(v):.4f} sd={v.std(ddof=1):.4f}  p={u8.pvalue:.3f}")

    # 条件内のシード分散
    e = load("expvar_seed")
    within = e.groupby("source")["f1"].std()
    between = e.groupby("source")["f1"].mean().std()
    print(f"\n  シード分散  条件内の標準偏差の平均={within.mean():.4f} / "
          f"条件平均の標準偏差={between:.4f}（比 {between / within.mean():.1f}倍）")
    print(e.groupby("source").agg(密度=("cond_density", "first"), n=("f1", "size"),
                                  平均=("f1", "mean"), 標準偏差=("f1", "std"),
                                  最小=("f1", "min"), 最大=("f1", "max")
                                  ).sort_values("密度").round(4).to_string())


def sec5_probe():
    head("第5節 UNet へ注入される残差")
    p = load("probe_residual")
    m = p[p["source"] != "__empty__"]
    r = stats.spearmanr(m["density"], m["total_rms"])
    dd = m.drop_duplicates(subset=["source", "total_rms"])
    r2 = stats.spearmanr(dd["density"], dd["total_rms"])
    print(f"  順伝播 {len(m)} 回        rho={r.correlation:+.3f} p={r.pvalue:.2g}")
    print(f"  地図とプロンプトの組 {len(dd)} 件  rho={r2.correlation:+.3f}")
    b = pd.cut(m["density"], [-1e-9, 1e-9, .005, .02, .05, 1],
               labels=["0", "〜0.005", "〜0.02", "〜0.05", "0.05〜"])
    print(m.groupby(b, observed=True).agg(n=("total_rms", "size"),
                                          総残差=("total_rms", "mean"),
                                          mid=("mid_rms", "mean")).round(4).to_string())
    z = m[m["density"] == 0]
    print(f"  密度0の45回は同一の全画素0地図。値は {z['total_rms'].nunique()} 通り"
          f"（{z['total_rms'].min():.4f}〜{z['total_rms'].max():.4f}、"
          f"sd={z['total_rms'].std():.4f}）でプロンプトの違いによる")

    inner = np.array([stats.spearmanr(g["density"], g["total_rms"]).correlation
                      for _, g in m.groupby("source")])
    print(f"  画像内の相関 n={len(inner)}  中央値={np.median(inner):+.3f}  "
          f"+0.98以上={int((inner >= .98).sum())}  すべて正={bool((inner > 0).all())}")

    e = load("exp6_depth_generality")
    ri = np.array([stats.spearmanr(g["cond_std"], g["total_rms"]).correlation
                   for _, g in e.groupby("source")])
    flat, orig = e[e["alpha"] == 0.0], e[e["alpha"] == 1.0]
    print(f"\n  depth 一般化  画像内の相関 n={len(ri)} 中央値={np.median(ri):+.3f} "
          f"正={int((ri > 0).sum())}/{len(ri)} 最小={ri.min():+.3f}")
    print(f"                平坦={flat['total_rms'].mean():.4f} "
          f"原画像={orig['total_rms'].mean():.4f} "
          f"（{orig['total_rms'].mean() / flat['total_rms'].mean():.2f}倍）")


def sec5_spatial():
    head("第5節 残差の空間構造（probe_spatial.py の出力）")
    f = os.path.join(R, "probe_spatial.csv")
    if not os.path.exists(f):
        return print("  probe_spatial.csv が無い。`python3 probe_spatial.py` を実行する")
    s = pd.read_csv(f)
    for r in s.itertuples():
        print(f"  {r.source:14s} 密度={r.density:.5f}  "
              f"mid RMS={r.mid_rms:.4f} 空間成分={r.mid_spatial_rms:.4f}"
              f"（{r.mid_spatial_frac * 100:.1f}%）  "
              f"down6 の空間成分={r.down6_spatial_frac * 100:.1f}%")
    print("  ※ 空間成分＝チャネルごとの空間平均を引いた残差の RMS。"
          "0 に近ければ一様な場、1 に近ければ空間的に変化している")


def sec6_improvement():
    head("第6節 改善前後（α層別・画像単位。プールした検定は使わない）")
    d = load("exp3_improvement")
    for a, g in d.groupby("alpha"):
        p = g.groupby("source").agg(before=("f1_before", "mean"), after=("f1_after", "mean"),
                                    db=("dens_before", "first"), da=("dens_after", "first"),
                                    sc=("scale_after", "first"))
        diff = p["after"] - p["before"]
        w = stats.wilcoxon(p["after"], p["before"])
        print(f"  α={a}  n={len(p)}画像  改善={int((diff > 0).sum())}/{len(p)}  "
              f"Δ中央値={diff.median():+.4f}  Wilcoxon p={w.pvalue:.4f}  "
              f"密度 {p['db'].mean():.4f}→{p['da'].mean():.4f}  scale平均={p['sc'].mean():.3f}")


def sec8_tradeoff():
    head("第8節 構造整合とプロンプト追従のトレードオフ")
    d = pd.concat([load("exp2_scale"), load("exp2b_scale_ext")],
                  ignore_index=True).dropna(subset=["clip", "f1"])
    print(d.groupby("scale").agg(n=("f1", "size"), edgeF1=("f1", "mean"),
                                 CLIP=("clip", "mean")).round(4).to_string())
    rf = np.array([stats.spearmanr(g["scale"], g["f1"]).correlation
                   for _, g in d.groupby("source")])
    rc = np.array([stats.spearmanr(g["scale"], g["clip"]).correlation
                   for _, g in d.groupby("source")])
    w = stats.wilcoxon(rc)
    print(f"\n  画像内の相関（n={len(rf)}画像。全70点は同一画像から10段階なので独立でない）")
    print(f"    scale-構造整合  中央値={np.median(rf):+.3f}  正={int((rf > 0).sum())}/{len(rf)}")
    print(f"    scale-プロンプト 中央値={np.median(rc):+.3f}  負={int((rc < 0).sum())}/{len(rc)}"
          f"  最大={rc.max():+.3f}  Wilcoxon符号順位検定 p={w.pvalue:.3f}")


def sec41_slicing():
    head("第4.1節 attention slicing と NaN")
    print(load("exp5_slicing").to_string(index=False))


if __name__ == "__main__":
    for fn in (sec3_env, sec41_slicing, sec42_density, sec5_probe,
               sec5_spatial, sec6_improvement, sec8_tradeoff):
        fn()
    print("\n" + "=" * 66)
    print("以上がレポート本文に記載した統計量のすべてである。")
