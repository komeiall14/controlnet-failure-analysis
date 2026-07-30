"""本文の数値が、手元のデータから出るものかを機械的に照合する。

レポートの誤りの多くは「確認していない数値を書いた」型だった。目視の点検では
見落とすので、本文中の数値を全部抜き出し、次のいずれかに当たるかを確かめる。

  1. stats_report.py の出力に現れる
  2. results/*.csv の値（そのもの、または一般的な丸め）に一致する
  3. 実験設定・引用など、データに由来しない既知の定数（下の ALLOW）

どれにも当たらない数値は「出所不明」として一覧に出す。出所不明がゼロになるまで
本文を直すか、根拠を stats_report.py から出せるようにするのが完了条件である。

    python3 audit_numbers.py
"""
import glob
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "REPORT.md")
R = os.path.join(HERE, "results")

# データに由来しない既知の定数。理由を必ず添える（書きっぱなしを防ぐため）
ALLOW = {
    "1.5": "Stable Diffusion v1.5 のバージョン",
    "2.2.2": "PyTorch のバージョン", "0.31.0": "diffusers のバージョン",
    "4.44.2": "transformers のバージョン", "4.11.0": "OpenCV のバージョン",
    "14.4.1": "macOS のバージョン", "2.0": "PyTorch 2.0 系（SDPA の文脈）",
    "512": "生成解像度", "64": "潜在変数の解像度／論文の重み式の分子",
    "20": "サンプラのステップ数", "16": "RAM(GB)", "8": "GPUコア数",
    "100": "Canny の低閾値", "200": "Canny の高閾値", "500": "測定に使う timestep",
    "0.045": "改善で狙う目標エッジ画素率（設計値）",
    "12": "二分探索の反復回数／論文のエンコーダブロック数",
    "400": "二分探索の探索上限", "2": "許容幅・分割数など",
    "0.15": "コントラスト係数", "0.3": "コントラスト係数", "0.5": "コントラスト係数",
    "1.0": "コントラスト係数／scale 既定値", "0.2": "scale 掃引点",
    "0.4": "scale 掃引点", "0.6": "scale 掃引点", "0.8": "scale 掃引点",
    "1.3": "scale 掃引点", "1.6": "scale 掃引点", "2.5": "scale 掃引点",
    "3.0": "scale 掃引点", "0.05": "有意水準／深度の係数",
    "50": "論文の prompt dropping 率(%)", "10": "論文の 12/1 ブロック・万枚など",
    "5": "シード数・条件数など", "3": "シード数など", "1": "汎用",
    "0": "汎用", "4": "DDIM ステップ数・分割数", "6": "1画像あたりの行数",
    "7": "入力数", "9": "掃引段数", "15": "画像数", "45": "条件数",
    "60": "条件数", "90": "対の数", "70": "点数", "135": "順伝播回数",
    "83": "重複除去後の件数", "68": "重複除去後の件数", "8": "独立枚数",
    "28": "論文の Figure 番号", "2.1": "supplementary の節番号",
    "4.2": "論文の節番号（アブレーション）", "3.4": "論文の節番号",
    "1.43": "モデルの総パラメータ数(×10^9)", "2.7": "fp16 の重みサイズ(GiB)",
    "5.3": "fp32 の重みサイズ(GiB)",
}


def report_numbers():
    """本文から数値を抜き出す。参考文献と節見出しは対象外。"""
    txt = open(SRC, encoding="utf-8").read()
    txt = txt.split("## 参考文献")[0]          # 書誌情報の年・巻号・ページは別管理
    out = []
    for i, line in enumerate(txt.split("\n"), 1):
        if line.startswith("#"):
            continue
        body = re.sub(r"`[^`]*`", "", line)     # コード片（識別子・既定値）は除く
        body = re.sub(r"\[\d+\]", "", body)     # 引用番号
        body = re.sub(r"第\s*[\d.]+\s*節", "", body)   # 節への参照
        body = re.sub(r"図\s*\d+", "", body)
        for m in re.finditer(r"\d+(?:\.\d+)?", body):
            out.append((i, m.group()))
    return out


def known_values():
    """stats_report.py の出力と results/*.csv から、根拠になりうる数値を集める。"""
    vals = set()

    r = subprocess.run([sys.executable, "stats_report.py"],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print("stats_report.py が失敗した:\n", r.stderr[-800:])
    for m in re.finditer(r"\d+(?:\.\d+)?", r.stdout):
        vals.add(m.group())
        # 本文は桁を落として引用することがあるので、丸めた表記も同一視する
        try:
            v = float(m.group())
        except ValueError:
            continue
        for nd in (1, 2, 3, 4):
            vals.add(f"{round(v, nd):.{nd}f}")

    for f in glob.glob(os.path.join(R, "*.csv")):
        d = pd.read_csv(f)
        for c in d.columns:
            s = pd.to_numeric(d[c], errors="coerce").dropna()
            for v in s:
                for nd in (0, 1, 2, 3, 4):
                    vals.add(f"{round(float(v), nd):.{nd}f}" if nd else str(int(round(v))))
                # 比・パーセント表記でも引けるように
                vals.add(f"{v * 100:.1f}")
    # 末尾の 0 を落とした表記も同一視する（0.50 と 0.5 など）
    vals |= {x.rstrip("0").rstrip(".") for x in list(vals) if "." in x}
    return vals


def main():
    known = known_values()
    unknown = []
    for line, n in report_numbers():
        if n in ALLOW or n in known or n.rstrip("0").rstrip(".") in known:
            continue
        unknown.append((line, n))

    if not unknown:
        print("出所不明の数値: なし")
        return 0

    print(f"出所不明の数値 {len(unknown)} 件（行番号: 値）")
    src = open(SRC, encoding="utf-8").read().split("\n")
    for line, n in unknown:
        ctx = src[line - 1]
        i = ctx.find(n)
        print(f"  L{line}: {n}\n        …{ctx[max(0, i - 40):i + 40]}…")
    return 1


if __name__ == "__main__":
    sys.exit(main())
