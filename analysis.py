"""実験結果の統計処理と作図。

図は本文に貼る前提で、日本語見出し・大きめの文字・出典注記つきにする。
使い方: python3 analysis.py [exp1|exp2|exp3|probe|all]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

for f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "AppleGothic"):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({"font.size": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 140})

NAVY, RED, GRAY = "#1f3864", "#c00000", "#808080"


def load(name):
    p = os.path.join(OUT, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def fig1_density():
    """失敗条件1: 条件地図の密度と構造整合。"""
    df = load("exp1_density")
    if df is None or len(df) < 4:
        return print("exp1: データ不足")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    syn = df[df["source"].str.startswith("synth")]
    real = df[~df["source"].str.startswith("synth")]
    ax[0].scatter(real["cond_density"], real["f1"], s=46, c=NAVY,
                  label="実画像 (skimage.data)", zorder=3)
    ax[0].scatter(syn["cond_density"], syn["f1"], s=46, c=RED, marker="^",
                  label="合成画像", zorder=3)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("条件地図のエッジ画素率（対数）")
    ax[0].set_ylabel("構造整合 edge F1")
    ax[0].set_title("固定閾値(100,200)では密度が\n下がると条件が効かなくなる")
    ax[0].legend(fontsize=10)
    r = stats.spearmanr(df["cond_density"], df["f1"])
    ax[0].text(0.03, 0.05, f"Spearman ρ={r.correlation:+.3f}\np={r.pvalue:.2g}",
               transform=ax[0].transAxes, fontsize=10,
               bbox=dict(fc="white", alpha=.8, ec=GRAY))

    for name, g in df.groupby("source"):
        g = g.sort_values("alpha")
        ax[1].plot(g["alpha"], g["f1"], "o-", alpha=.65, lw=1.4, ms=4)
    ax[1].set_xlabel("入力コントラスト係数 α（構図は不変）")
    ax[1].set_ylabel("構造整合 edge F1")
    ax[1].set_title("同一画像でコントラストだけを落とすと\n構造整合が崩れる")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_density.png"), bbox_inches="tight")
    print(f"  fig1: n={len(df)} ρ={r.correlation:+.3f} p={r.pvalue:.3g}")


def fig2_scale():
    """失敗条件2: 推論時の既定 scale が構造整合に対して弱い。

    当初は「最適値が入力の密度に依存する」と予想したが外れた（本文第4.3節）。
    構造整合だけで選ぶと全入力で掃引範囲の上限が最良になる。
    """
    df = load("exp2_scale")
    if df is None or len(df) < 4:
        return print("exp2: データ不足")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    best = []
    for name, g in df.groupby("source"):
        g = g.sort_values("scale")
        ax[0].plot(g["scale"], g["f1"], "o-", lw=1.5, ms=5, label=name[:14])
        b = g.loc[g["f1"].idxmax()]
        best.append(dict(source=name, density=b["cond_density"],
                         best_scale=b["scale"], best_f1=b["f1"]))
    ax[0].axvline(1.0, color=GRAY, ls="--", lw=1)
    ax[0].text(1.02, ax[0].get_ylim()[0], "既定値 1.0", color=GRAY, fontsize=9)
    ax[0].set_xlabel("controlnet_conditioning_scale")
    ax[0].set_ylabel("構造整合 edge F1")
    ax[0].set_title("既定 1.0 は構造整合に対して弱い")
    ax[0].legend(fontsize=8, ncol=2)

    b = pd.DataFrame(best)
    ax[1].scatter(b["density"], b["best_scale"], s=70, c=RED, zorder=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("条件地図のエッジ画素率（対数）")
    ax[1].set_ylabel("F1 を最大化する scale")
    ax[1].set_title("密度と scale は同じ資源の\n二つの支払い方になっている")
    if len(b) > 2:
        r = stats.spearmanr(b["density"], b["best_scale"])
        ax[1].text(0.03, 0.9, f"Spearman ρ={r.correlation:+.3f}",
                   transform=ax[1].transAxes, fontsize=10,
                   bbox=dict(fc="white", alpha=.8, ec=GRAY))
        print(f"  fig2: 密度 vs 最適scale ρ={r.correlation:+.3f} p={r.pvalue:.3g}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_scale.png"), bbox_inches="tight")
    b.to_csv(os.path.join(OUT, "exp2_best_scale.csv"), index=False)


def fig3_improvement():
    """改善前後の対応比較（参考図）。

    ★注意: ここでプールしている Wilcoxon は、1画像あたり6行を独立扱いするため
    標本数が水増しされる。レポート第6節の結論はこの検定ではなく、
    stats_report.py が出す α 層別・画像単位の検定に基づく。この図は分布の概観用。
    """
    df = load("exp3_improvement")
    if df is None or len(df) < 6:
        return print("exp3: データ不足")
    d = df["f1_after"] - df["f1_before"]
    w = stats.wilcoxon(df["f1_after"], df["f1_before"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(df["f1_before"], df["f1_after"], s=42, c=NAVY, zorder=3)
    lim = [0, max(df[["f1_before", "f1_after"]].max()) * 1.08]
    ax[0].plot(lim, lim, "--", c=GRAY, lw=1)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("改善前 edge F1（固定閾値・scale 1.0）")
    ax[0].set_ylabel("改善後 edge F1（密度適応・scale 連動）")
    ax[0].set_title("同一 seed・同一プロンプトの対応比較")
    ax[0].text(0.03, 0.88,
               f"n={len(df)}  改善 {(d>0).sum()}/{len(d)}\n"
               f"中央値 Δ={d.median():+.3f}\nWilcoxon p={w.pvalue:.3g}",
               transform=ax[0].transAxes, fontsize=10,
               bbox=dict(fc="white", alpha=.85, ec=GRAY))
    lowc = df[df["alpha"] < 1.0]
    highc = df[df["alpha"] >= 1.0]
    ax[1].boxplot([highc["f1_before"], highc["f1_after"],
                   lowc["f1_before"], lowc["f1_after"]],
                  labels=["原画像\n前", "原画像\n後", "低コントラスト\n前",
                          "低コントラスト\n後"])
    ax[1].set_ylabel("構造整合 edge F1")
    ax[1].set_title("破綻していた低コントラスト側ほど\n改善幅が大きい")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_improvement.png"), bbox_inches="tight")
    print(f"  fig3: n={len(df)} 改善{(d>0).sum()}/{len(d)} "
          f"Δ中央値={d.median():+.4f} Wilcoxon p={w.pvalue:.4g}")
    if len(lowc):
        dl = lowc["f1_after"] - lowc["f1_before"]
        print(f"        低コントラストのみ: n={len(lowc)} Δ中央値={dl.median():+.4f}")


def fig4_probe():
    """機構の直接証拠: 密度と ControlNet 残差ノルム。"""
    df = load("probe_residual")
    if df is None:
        return print("probe: データ不足")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    m = df[df["source"] != "__empty__"]
    pos = m[m["density"] > 0]
    zero = m[m["density"] <= 0]
    # 密度0は対数軸に載らないので、正の最小値の1/3の位置に別枠として置く
    zx = pos["density"].min() / 3 if len(pos) else 1e-4
    ax.scatter(pos["density"], pos["total_rms"], s=30, c=NAVY, alpha=.75,
               zorder=3, label="エッジあり")
    if len(zero):
        ax.scatter([zx] * len(zero), zero["total_rms"], s=34, c=RED,
                   marker="x", zorder=3, label=f"エッジ画素率0 (n={len(zero)})")
        ax.axvline(zx * 1.8, color=GRAY, ls=":", lw=1)
    # ★空白地図の基準線は描かない。
    # 密度0の条件地図は全画素0で、参照に使った空白地図と同一の条件地図である。
    # 密度を動かしていない比較なので、値が近くても密度の効果については
    # 何も語らない（本文第5節で撤回済み）。
    ax.set_xscale("log")
    ax.set_xlabel("条件地図のエッジ画素率（対数。左端の×は画素率0）")
    ax.set_ylabel("UNet へ注入される残差の RMS")
    ax.set_title("密度が下がると zero-convolution 経由の\n残差そのものが縮む")
    ax.legend(fontsize=9, loc="lower right")
    r = stats.spearmanr(m["density"], m["total_rms"])
    dd = m.drop_duplicates(subset=["source", "total_rms"])
    r2 = stats.spearmanr(dd["density"], dd["total_rms"])
    # 本文が機構の根拠に据えているのは密度>0 の領域なので、そちらも併記する
    nzp = m[m["density"] > 0]
    r3 = stats.spearmanr(nzp["density"], nzp["total_rms"])
    ax.text(0.03, 0.87,
            f"Spearman ρ={r.correlation:+.3f}\n重複除去 ρ={r2.correlation:+.3f}\n"
            f"密度>0のみ ρ={r3.correlation:+.3f}",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(fc="white", alpha=.85, ec=GRAY))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_residual.png"), bbox_inches="tight")
    print(f"  fig4: n={len(m)} ρ={r.correlation:+.3f} p={r.pvalue:.3g}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = dict(exp1=fig1_density, exp2=fig2_scale, exp3=fig3_improvement,
               probe=fig4_probe)
    for k, fn in fns.items():
        if which in ("all", k):
            fn()
