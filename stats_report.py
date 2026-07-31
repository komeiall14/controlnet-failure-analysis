"""レポート本文に載せた統計量を再現する。

analysis.py は図を描くためのもので、fig1・fig2・fig3 には本文が採らなかった扱い
（定義上 0 になる条件を含む相関、1画像6行をプールした検定）が残っている。
本文の数値の出どころはこのスクリプトである。

    python3 stats_report.py
"""
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
    # 対象は名指しで指定する。「sec 列を持つ CSV を全部」のような集め方だと、
    # DDIM 4 ステップで測る exp5・bench_dtype が混ざって別条件の値が入る。
    # 速度の母集団は run_one() が回した 20 ステップの生成だけである。
    SPEED = ("exp1_density", "exp2_scale", "exp2b_scale_ext",
             "expvar_seed", "smoke")
    s = []
    for n in SPEED:
        f = os.path.join(R, f"{n}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if "sec" in d.columns:
            s += d["sec"].dropna().tolist()
    s = np.array(s)
    print(f"  n={len(s)}  min={s.min():.1f}  max={s.max():.1f}  "
          f"中央値={np.median(s):.1f}  平均={s.mean():.1f}")


def sec41_cause():
    head("第4.1節 NaN の原因の切り分け（exp8_slicing_cause.py の出力）")
    f = os.path.join(R, "exp8_slicing_cause.csv")
    if not os.path.exists(f):
        return print("  exp8_slicing_cause.csv が無い。`python3 exp8_slicing_cause.py` を実行する")
    d = pd.read_csv(f)
    print(d.to_string(index=False))
    bad = d[d["latent_nan"] & d["first_nan_call"].notna()]
    if len(bad):
        print(f"  NaN が出た条件での入力の絶対値の最大 "
              f"{max(bad['q_absmax'].max(), bad['k_absmax'].max()):.3f}"
              f"（fp16 の上限 65504）")
        print(f"  最初に NaN を出した呼び出し（0 起点の添字）"
              f"{sorted(set(bad['first_nan_call'].astype(int)))}")
    ok = d[(d["buffer"] == "zeros") & (~d["latent_nan"])]
    if len(ok):
        # 本文が挙げる範囲は seed 0 で粒度 1・4・auto を比べたもの。
        # seed を変えた条件まで混ぜると別の範囲になり、本文と対応しなくなる。
        s0 = ok[ok["seed"] == 0] if "seed" in ok.columns else ok
        print(f"  加算元を zeros にした {len(ok)} 条件はすべて正常。"
              f"seed 0 で粒度を変えた {len(s0)} 条件の潜在変数の絶対値の最大は "
              f"{s0['latent_absmax'].min():.4f}〜{s0['latent_absmax'].max():.4f}")
        if len(s0) < len(ok):
            print(f"  （seed を変えた {len(ok) - len(s0)} 条件も正常。"
                  f"そちらを含めた範囲は {ok['latent_absmax'].min():.4f}〜"
                  f"{ok['latent_absmax'].max():.4f}）")


def sec41_buffer():
    head("第4.1節 加算元バッファの中身（exp10_buffer_content.py の出力）")
    f = os.path.join(R, "exp10_buffer_content.csv")
    if not os.path.exists(f):
        return print("  exp10_buffer_content.csv が無い。`python3 exp10_buffer_content.py` を実行する")
    d = pd.read_csv(f)
    for r in d.itertuples():
        print(f"  {r.dtype}: 呼び出し {r.calls}  潜在変数NaN={r.latent_nan}  "
              f"非有限を含むバッファ {r.buffers_with_nonfinite}/{r.calls}  "
              f"非有限要素 {r.nonfinite_elements}/{r.elements_checked}"
              f"（{r.nonfinite_ratio * 100:.1f}%）")
        print(f"      最初に非有限が出た呼び出し {r.first_dirty_call} / "
              f"最初に出力が NaN になった呼び出し {r.first_nan_out_call}")


def sec41_beta():
    head("第4.1節 baddbmm が beta=0 で加算元を無視するか（exp11_beta_zero.py）")
    f = os.path.join(R, "exp11_beta_zero.csv")
    if not os.path.exists(f):
        return print("  exp11_beta_zero.csv が無い。`python3 exp11_beta_zero.py` を実行する")
    d = pd.read_csv(f)
    print(d.to_string(index=False))
    for dev, g in d.groupby("device"):
        print(f"  {dev}: 規約どおり {int(g['honors_beta0'].sum())}/{len(g)} 条件")


def sec41_fix():
    head("第4.1節 修正が実用になるか（exp12_fix_usable.py、20ステップ）")
    f = os.path.join(R, "exp12_fix_usable.csv")
    if not os.path.exists(f):
        return print("  exp12_fix_usable.csv が無い。`python3 exp12_fix_usable.py` を実行する")
    d = pd.read_csv(f)
    print(d.groupby("cond").agg(壊れた枚数=("broken", "sum"), n=("broken", "size"),
                                edgeF1平均=("f1", "mean")).round(4).to_string())
    g = os.path.join(R, "exp12_pixel_diff.csv")
    if os.path.exists(g):
        t = pd.read_csv(g)
        print(f"  修正版と slicing 無効時の画素差（0〜255、{len(t)}枚）"
              f"平均 {t['mean_abs_diff'].mean():.2f}  最大 {t['max_abs_diff'].max():.0f}")


def sec41_mem():
    head("第4.1節 修正が省メモリ効果を保つか（exp13_memory.py）")
    f = os.path.join(R, "exp13_memory.csv")
    if not os.path.exists(f):
        return print("  exp13_memory.csv が無い。`python3 exp13_memory.py` を実行する")
    d = pd.read_csv(f)
    print(d.to_string(index=False))
    a_ = dict(zip(d["cond"], d["peak_gib"]))
    if "slicingなし" in a_ and "slicingあり修正" in a_:
        g = a_["slicingなし"] - a_["slicingあり修正"]
        print(f"  修正版は slicing 無効時より {g:+.3f} GiB"
              f"（{g / a_['slicingなし'] * 100:+.1f}%）。"
              f"壊れたまま使う slicingあり と同じ {a_['slicingあり修正']:.3f} GiB")


def sec41_bench():
    head("第4.1節 fp16 と fp32 の生成時間（ウォームアップ1回を捨てて3回）")
    f = os.path.join(R, "bench_dtype.csv")
    if not os.path.exists(f):
        return print("  bench_dtype.csv が無い。`python3 bench_dtype.py` を実行する")
    d = pd.read_csv(f)
    g = d.groupby("dtype")["sec"].agg(n="size", 中央値="median", 最小="min", 最大="max")
    print(g.round(2).to_string())
    print(f"  比 fp32/fp16 = {g.loc['fp32', '中央値'] / g.loc['fp16', '中央値']:.2f}倍")


def sec42_density():
    head("第4.2節 条件地図の密度と構造整合")
    d = load("exp1_density")
    r_all = stats.spearmanr(d["cond_density"], d["f1"])
    nz = d[d["cond_density"] > 0]
    r_nz = stats.spearmanr(nz["cond_density"], nz["f1"])
    print(f"  全60条件            rho={r_all.correlation:+.3f} p={r_all.pvalue:.2g}")
    # 45条件は1画像から最大4点出るので独立でない。本文はこの断りを添えて出す。
    dd = nz.drop_duplicates(subset=["source", "cond_density", "f1"])
    r_dd = stats.spearmanr(dd["cond_density"], dd["f1"])
    inner = [(src, stats.spearmanr(g["cond_density"], g["f1"]).correlation)
             for src, g in nz.groupby("source") if g["cond_density"].nunique() >= 3]
    print(f"  密度>0・重複除去 {len(dd)}件  rho={r_dd.correlation:+.3f} p={r_dd.pvalue:.2g}")
    print(f"  画像内（密度が3値以上動く {len(inner)}画像）  "
          f"正 {sum(r > 0 for _, r in inner)}/{len(inner)}  "
          f"{', '.join(f'{n}={r:+.2f}' for n, r in inner)}")
    print(f"  密度>0の45条件のみ  rho={r_nz.correlation:+.3f} p={r_nz.pvalue:.2g}")
    print("  ※ 条件地図が空だと edge F1 は定義上 0 を返すので、前者は押し上げられている")

    b = pd.cut(d["cond_density"], [-1e-9, 1e-9, .005, .02, .05, 1],
               labels=["0", "〜0.005", "〜0.02", "〜0.05", "0.05〜"])
    print("\n  密度帯ごと（密度0の行は指標が使えないので参考値）")
    print(d.groupby(b, observed=True)["f1"]
          .agg(n="size", 平均="mean", 中央値="median", 標準偏差="std").round(4).to_string())

    # 条件が効いていないことの証拠：本来指定したかった輪郭との一致。
    # 生成画像があれば画像から直接計算し、無ければ image_metrics.csv を使う
    # （画像は 163MB あるためリポジトリに含めていない）。
    tbl = {n: im for n, im, _ in ds.real_images() + ds.synthetic_images()}
    have_images = os.path.exists(os.path.join(R, "images", f"{d.iloc[0]['tag']}_gen.png"))
    z = d[d["cond_density"] == 0]
    ok = d[d["alpha"] == 1.0]

    if have_images:
        print("  （出所: 生成画像から直接計算）")

        def f1_vs_target(tag, src):
            g = cv2.imread(os.path.join(R, "images", f"{tag}_gen.png"))
            return None if g is None else P.edge_f1(P.canny(tbl[src]), g)["f1"]

        zs = [x for x in (f1_vs_target(r.tag, r.source) for r in z.itertuples()) if x is not None]
        os_ = [x for x in (f1_vs_target(r.tag, r.source) for r in ok.itertuples()) if x is not None]
        floor = []
        for r in ok.itertuples():
            g = cv2.imread(os.path.join(R, "images", f"{r.tag}_gen.png"))
            if g is None:
                continue
            for other in tbl:
                if other != r.source:
                    floor.append(P.edge_f1(P.canny(tbl[other]), g)["f1"])
        md5s = {}
        for r in z.itertuples():
            f = os.path.join(R, "images", f"{r.tag}_gen.png")
            if os.path.exists(f):
                md5s[r.tag] = hashlib.md5(open(f, "rb").read()).hexdigest()
    else:
        print("  （出所: results/image_metrics.csv。生成画像が無い環境向けのキャッシュ）")
        im = pd.read_csv(os.path.join(R, "image_metrics.csv"))
        own = im[im["source"] == im["ref_source"]]
        zs = own[own["cond_density"] == 0]["f1"].tolist()
        os_ = own[own["alpha"] == 1.0]["f1"].tolist()
        floor = im[(im["alpha"] == 1.0) & (im["source"] != im["ref_source"])]["f1"].tolist()
        md5s = dict(im[im["cond_density"] == 0][["tag", "md5"]].drop_duplicates().values)

    def f1_of(tag):
        """md5 重複を除くときに使う、タグ→自分の輪郭との F1。"""
        if have_images:
            r = d[d["tag"] == tag].iloc[0]
            g = cv2.imread(os.path.join(R, "images", f"{tag}_gen.png"))
            return None if g is None else P.edge_f1(P.canny(tbl[r["source"]]), g)["f1"]
        im = pd.read_csv(os.path.join(R, "image_metrics.csv"))
        q = im[(im["tag"] == tag) & (im["source"] == im["ref_source"])]
        return float(q["f1"].iloc[0]) if len(q) else None
    u = stats.mannwhitneyu(zs, floor)
    print(f"\n  本来の輪郭との一致  正しく条件づけ={np.mean(os_):.4f} (n={len(os_)})")
    print(f"                      密度0        ={np.mean(zs):.4f} (n={len(zs)}"
          f"、中央値={np.median(zs):.4f}、sd={np.std(zs, ddof=1):.4f})")
    print(f"                      偶然の水準    ={np.mean(floor):.4f} (n={len(floor)})")
    print(f"                      Mann-Whitney p={u.pvalue:.3f}")

    # md5 重複を除いた独立な8枚
    seen = {}
    for tag, h in md5s.items():
        if h not in seen:
            seen[h] = f1_of(tag)
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
    # 密度0の45回を除いた内訳。本文はこの落差も併記している
    nz = m[m["density"] > 0]
    nzd = nz.drop_duplicates(subset=["source", "total_rms"])
    print(f"  密度>0のみ {len(nz)} 回   rho={stats.spearmanr(nz['density'], nz['total_rms']).correlation:+.3f}"
          f"  / 重複除去 {len(nzd)} 件 rho="
          f"{stats.spearmanr(nzd['density'], nzd['total_rms']).correlation:+.3f}")
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
    print(f"  画像内の相関（密度0を含む） n={len(inner)}  中央値={np.median(inner):+.3f}  "
          f"+0.98以上={int((inner >= .98).sum())}  すべて正={bool((inner > 0).all())}")
    # 密度0の点は「密度を動かしていない比較」なので本節では使わない。
    # 密度が正の範囲に限り、さらに密度が実際に段階的に動く画像だけを見る。
    var = []
    for src, g in m.groupby("source"):
        g = g[g["density"] > 0]
        if g["density"].nunique() < 3 or g["density"].max() / g["density"].min() <= 1.5:
            continue
        var.append((src, stats.spearmanr(g["density"], g["total_rms"]).correlation))
    if var:
        v = np.array([x[1] for x in var])
        print(f"  画像内の相関（密度>0 かつ密度が段階的に動く画像のみ） {len(var)}枚  "
              f"{v.min():+.3f}〜{v.max():+.3f}  中央値={np.median(v):+.3f}")
        print(f"    対象: {', '.join(x[0] for x in var)}")
        print(f"    除いた {len(inner) - len(var)}枚は、正の密度が3値未満か"
              f"変化が1.5倍以下で系列内の勾配が取れない")

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
        # 条件地図が空だった画像だけを取り出す（本文はこの回復幅を別に述べている）
        emp = p[p["db"] == 0]
        if len(emp):
            de = emp["after"] - emp["before"]
            print(f"        うち改善前の密度が0だった {len(emp)}画像  "
                  f"Δ中央値={de.median():+.4f}  密度 0→{emp['da'].mean():.4f}")
        # 個々の画像の扱いで結論が動くかを見る感度分析
        for drop in (["synth_veryd"], ["cell", "chelsea"]):
            q = p.drop([x for x in drop if x in p.index], errors="ignore")
            if len(q) == len(p) or len(q) < 3:
                continue
            dq = q["after"] - q["before"]
            wq = stats.wilcoxon(q["after"], q["before"])
            print(f"        {'+'.join(drop)} を除く: n={len(q)}  "
                  f"改善={int((dq > 0).sum())}/{len(q)}  Δ中央値={dq.median():+.4f}  "
                  f"p={wq.pvalue:.4f}")


def sec6_common_ref():
    head("第6節 共通の参照地図で測り直した改善前後")
    print("  exp3 の f1_before/f1_after は、それぞれが与えられた条件地図に対する値である。"
          "\n  地図の密度が違えば F1 も動く（第4.2節）ので、両者を同じ参照"
          "（本来指定したかった輪郭）で測り直す。")
    tbl = {n: im for n, im, _ in ds.real_images() + ds.synthetic_images()}
    d = load("exp3_improvement")
    # 生成画像が無い環境では imread が1行ごとに警告を出すので、先に1回だけ確かめる
    r0 = d.iloc[0]
    probe = os.path.join(R, "images",
                         f"e3_{r0['source']}_a{r0['alpha']}_s{r0['seed']}_before.png")
    if not os.path.exists(probe):
        # 生成画像が無い環境でも同じ値が出せる。exp7_policy.csv は同じ 90 対に
        # ついて共通参照での F1（f1c_before / f1c_after）を持っている。
        f = os.path.join(R, "exp7_policy.csv")
        if not os.path.exists(f):
            return print("  生成画像も exp7_policy.csv も無いので測り直せない")
        print("  （出所: exp7_policy.csv。生成画像から直接計算した値と一致する）")
        c = pd.read_csv(f)
        for a_, g in c.groupby("alpha"):
            t = g.groupby("source").agg(b=("f1c_before", "mean"),
                                        a2=("f1c_after", "mean"))
            d2 = t["a2"] - t["b"]
            w = stats.wilcoxon(t["a2"], t["b"])
            print(f"  α={a_}  n={len(t)}画像  改善={int((d2 > 0).sum())}/{len(t)}  "
                  f"Δ中央値={d2.median():+.4f}  Wilcoxon p={w.pvalue:.4f}")
        return
    rows = []
    for r in d.itertuples():
        ref = P.canny(tbl[r.source])
        gb = cv2.imread(os.path.join(R, "images",
                                     f"e3_{r.source}_a{r.alpha}_s{r.seed}_before.png"))
        ga = cv2.imread(os.path.join(R, "images",
                                     f"e3_{r.source}_a{r.alpha}_s{r.seed}_after.png"))
        if gb is None or ga is None:
            continue
        rows.append(dict(source=r.source, alpha=r.alpha,
                         b=P.edge_f1(ref, gb)["f1"], a=P.edge_f1(ref, ga)["f1"]))
    if rows:
        print("  （出所: 生成画像から直接計算）")
        c = pd.DataFrame(rows)
    else:
        # 画像が無い環境では exp7_policy.csv の同じ値を使う
        print("  （出所: results/exp7_policy.csv。生成画像が無い環境向け）")
        e = load("exp7_policy")
        c = e.rename(columns={"f1c_before": "b", "f1c_after": "a"})[
            ["source", "alpha", "b", "a"]]
    for al, g in c.groupby("alpha"):
        p = g.groupby("source").agg(b=("b", "mean"), a=("a", "mean"))
        diff = p["a"] - p["b"]
        w = stats.wilcoxon(p["a"], p["b"])
        print(f"  α={al}  n={len(p)}画像  改善={int((diff > 0).sum())}/{len(p)}  "
              f"Δ中央値={diff.median():+.4f}  Wilcoxon p={w.pvalue:.4f}")


def sec6_policy():
    head("第6節 プロンプト追従と、推奨形（条件付き適用）の評価")
    f = os.path.join(R, "exp7_policy.csv")
    if not os.path.exists(f):
        return print("  exp7_policy.csv が無い。`python3 exp7_policy.py` を実行する")
    c = pd.read_csv(f)
    for a, g in c.groupby("alpha"):
        p = g.groupby("source").agg(db=("dens_before", "first"),
                                    f1b=("f1_before", "mean"), f1a=("f1_after", "mean"),
                                    fcb=("f1c_before", "mean"), fca=("f1c_after", "mean"),
                                    cb=("clip_before", "mean"), ca=("clip_after", "mean"))
        wc = stats.wilcoxon(p["ca"], p["cb"])
        print(f"  α={a}  CLIP {p['cb'].mean():.4f}→{p['ca'].mean():.4f}  "
              f"Δ中央値={(p['ca'] - p['cb']).median():+.4f}  p={wc.pvalue:.4f}")
        use = p["db"] < 0.045
        for lab, aft, bef in (("各自の地図", p["f1a"], p["f1b"]),
                              ("共通参照", p["fca"], p["fcb"])):
            pol = np.where(use, aft, bef)
            w = stats.wilcoxon(pol, bef)
            print(f"      推奨形({lab}) 適用{int(use.sum())}/{len(p)}  "
                  f"Δ中央値={np.median(pol - bef):+.4f}  p={w.pvalue:.4f}")


def sec6_holdout():
    head("第6節 開発に使っていない画像での検証（held-out）")
    f = os.path.join(R, "exp9_holdout.csv")
    if not os.path.exists(f):
        return print("  exp9_holdout.csv が無い。`python3 exp9_holdout.py` を実行する")
    c = pd.read_csv(f)
    # skimage.data.cat は chelsea のエイリアスで開発側と同一画像なので held-out から外す
    c = c[c["source"] != "cat"]
    done = c.groupby("source").size()
    if len(done) < len(ds.holdout_images()):
        print(f"  ※ 実行途中（{len(c)} 行、{len(done)} 画像）。以下は暫定値")
    print("  改善と適用条件は real_images 10点＋合成5点から作った。"
          "ここでは規則を一切変えずに当てている。")
    for a, g in c.groupby("alpha"):
        p_ = g.groupby("source").agg(db=("dens_before", "first"),
                                     b=("f1_before", "mean"), a_=("f1_after", "mean"),
                                     cb=("f1c_before", "mean"), ca=("f1c_after", "mean"))
        if len(p_) < 3:
            print(f"  α={a}  n={len(p_)}画像（検定には足りない）")
            continue
        for lab, aft, bef in (("各自の地図", p_["a_"], p_["b"]),
                              ("共通参照", p_["ca"], p_["cb"])):
            d = aft - bef
            w = stats.wilcoxon(aft, bef)
            print(f"  α={a} 無条件適用({lab})  n={len(p_)}  "
                  f"改善={int((d > 0).sum())}/{len(p_)}  Δ中央値={d.median():+.4f}  "
                  f"p={w.pvalue:.4f}")
        use = p_["db"] < 0.045
        for lab, aft, bef in (("各自の地図", p_["a_"], p_["b"]),
                              ("共通参照", p_["ca"], p_["cb"])):
            pol = np.where(use, aft, bef)
            w = stats.wilcoxon(pol, bef)
            print(f"  α={a} 推奨形({lab})      適用{int(use.sum())}/{len(p_)}  "
                  f"Δ中央値={np.median(pol - bef):+.4f}  p={w.pvalue:.4f}")
    dev = {n for n, _, _ in ds.real_images() + ds.synthetic_images()}
    dup = dev & set(c["source"].unique())
    print(f"  開発側と重複する画像: {sorted(dup) if dup else 'なし'}"
          f"（held-out {c['source'].nunique()} 画像）")
    z = c[(c["dens_before"] == 0)]
    if len(z):
        zz = z.groupby("source").agg(cb=("f1c_before", "mean"), ca=("f1c_after", "mean"),
                                     da=("dens_after", "first"))
        print(f"\n  改善前に条件地図が空だった {len(zz)} 画像  "
              f"共通参照 {zz['cb'].mean():.4f} → {zz['ca'].mean():.4f}  "
              f"密度 0→{zz['da'].mean():.4f}")


def sec7_conflict():
    head("（本文には載せていない予備実験）条件とプロンプトの衝突")
    d = load("exp4_conflict")
    g = d.groupby("kind")["f1"].agg(n="size", 平均="mean", 中央値="median").round(4)
    print(g.to_string())
    c = d[d["kind"] != "match"].groupby("kind")["f1"].mean()
    print(f"  一致条件の平均={d[d['kind'] == 'match']['f1'].mean():.4f}  "
          f"衝突条件の平均は {c.min():.4f}〜{c.max():.4f} に分布")
    pv = d.pivot_table(index="source", columns="kind", values="f1")
    fr = stats.friedmanchisquare(pv["match"], pv["conflict0"],
                                 pv["conflict1"], pv["conflict2"])
    print(f"  4条件をまたぐ差 Friedman p={fr.pvalue:.4f}（n={len(pv)}画像）")
    for k in ("conflict0", "conflict1", "conflict2"):
        w = stats.wilcoxon(pv["match"], pv[k])
        print(f"    match vs {k}: Δ中央値={(pv[k] - pv['match']).median():+.4f}"
              f"  p={w.pvalue:.4f}")


def sec8_amplify():
    head("第8節 空の条件地図で scale を上げても指定が戻らないか")
    f = os.path.join(R, "exp14_amplify_empty.csv")
    if not os.path.exists(f):
        return print("  exp14_amplify_empty.csv が無い。"
                     "`python3 exp14_amplify_empty.py` を実行する")
    d = pd.read_csv(f)
    for (src, a_), g in d.groupby(["source", "alpha"]):
        g = g.sort_values("scale")
        v = g["f1_vs_intended"].values
        lab = "空の地図" if g["cond_density"].iloc[0] == 0 else "条件地図あり"
        print(f"  {src:8s} α={a_:<5}（{lab}） scale {g['scale'].min()}→"
              f"{g['scale'].max()}  本来の輪郭との F1 {v[0]:.4f}→{v[-1]:.4f}"
              f"  幅 {v.max() - v.min():.4f}")


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

    # 最適 scale が密度で動くかどうか（第4.3節の予想の検証）
    rows = []
    for src, g in d.groupby("source"):
        a = g.groupby("scale").agg(f1=("f1", "mean"), clip=("clip", "mean"),
                                   dens=("cond_density", "first"))
        def norm(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
        both = norm(a["f1"]) + norm(a["clip"])
        rows.append(dict(source=src, dens=a["dens"].iloc[0],
                         best_f1=a["f1"].idxmax(), best_both=both.idxmax()))
    t = pd.DataFrame(rows)
    for lab, col in (("構造整合だけ", "best_f1"), ("両指標の和", "best_both")):
        r = stats.spearmanr(t["dens"], t[col])
        print(f"    最適scaleと密度（{lab}） rho={r.correlation:+.3f} p={r.pvalue:.2f}"
              f"  範囲 {t[col].min():g}〜{t[col].max():g}")
    # 既定値 1.0 が両指標の和の平均でどこに来るか
    m2 = d.groupby(["source", "scale"]).agg(f1=("f1", "mean"), clip=("clip", "mean"))
    per = []
    for src, g in m2.groupby(level=0):
        g = g.droplevel(0)
        def norm(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
        per.append(norm(g["f1"]) + norm(g["clip"]))
    avg = pd.concat(per, axis=1).mean(axis=1).sort_values(ascending=False)
    print(f"    両指標の和の画像平均が最大の scale = {avg.index[0]:g}"
          f"（{avg.iloc[0]:.3f}）／次点 {avg.index[1]:g}（{avg.iloc[1]:.3f}）")


def sec41_slicing():
    head("第4.1節 attention slicing と NaN")
    print(load("exp5_slicing").to_string(index=False))


if __name__ == "__main__":
    for fn in (sec3_env, sec41_slicing, sec41_cause, sec41_buffer, sec41_beta,
               sec41_fix, sec41_mem, sec41_bench, sec42_density, sec5_probe, sec5_spatial,
               sec6_improvement, sec6_common_ref, sec6_policy, sec6_holdout,
               sec8_amplify, sec8_tradeoff):
        fn()
    print("\n" + "=" * 66)
    print("以上がレポート本文に記載した統計量のすべてである。")
    print("以下は本文に採らなかった予備実験で、本文に対応する記述は無い。")
    sec7_conflict()
