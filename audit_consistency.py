"""提出物どうしの一貫性を点検する。

提出物はレポート（PDF）とソースコードの両方である。本文だけを直して
コードの docstring や README を放置すると、リポジトリを開いた時点で
食い違いが見える。画像数のように同じ数が四つのファイルに散っている
ものは目で追えないので、実体から取って機械で照合する。

見るのは4つ。

  A 画像数     コードが返す実体と、本文・README・LICENSE・docstring の記述
  B 共通の数値 README と REPORT.md の両方に出る数値が一致するか
  C ファイル   README が挙げるものが実在し、実在するものが漏れていないか
  D 名指し     本文が名前を挙げたスクリプト・CSV が実在するか

    python3 audit_consistency.py
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    p = os.path.join(HERE, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def counts_in(txt, word):
    """「〜 N 画像」「N 点」のような記述を拾う。"""
    out = set()
    for m in re.finditer(rf"(\d+)\s*(?:画像|点|枚){{0,1}}(?=[^0-9]|$)", txt):
        pass
    for m in re.finditer(rf"{word}[^。\n]{{0,24}}?(\d+)\s*(?:画像|点|枚)", txt):
        out.add(int(m.group(1)))
    return out


def check_images():
    """A: 開発側と held-out の画像数が、実体と記述で一致するか。"""
    bad = []
    try:
        import dataset as ds
        dev = len(ds.real_images()) + len(ds.synthetic_images())
        hold = len(ds.holdout_images())
    except Exception as e:
        return [("dataset.py が読めない", str(e))]

    rep = read("REPORT.md")
    # 本文が held-out について述べる画像数
    m = re.search(r"使っていない[^。]{0,20}?(\d+)\s*画像", rep)
    if m and int(m.group(1)) != hold:
        bad.append((f"held-out の画像数", f"本文 {m.group(1)} 対 実体 {hold}"))
    # 開発側
    m = re.search(r"(\d+)\s*画像 × コントラスト", rep)
    if m and int(m.group(1)) != dev - 0:
        bad.append(("開発側の画像数", f"本文 {m.group(1)} 対 実体 {dev}"))

    # docstring と LICENSE
    # 照合語は短く取る。「開発に使っていない」で探していたときは、実文の
    # 「開発に一切使っていない 8 点」に「一切」が挟まって一致せず、
    # まさにこの検査が動機に挙げている食い違いを素通りさせていた。
    for f, word in (("dataset.py", "使っていない"),
                    ("exp9_holdout.py", "使っていない"),
                    ("LICENSE_DATA.md", "取り分けた")):
        t = read(f)
        for n in counts_in(t, word):
            if n != hold:
                bad.append((f"{f} の記述", f"{n} 対 実体 {hold}"))
    return bad


def check_shared_numbers():
    """B: README と REPORT.md の両方に出る特徴的な数値が一致するか。"""
    rep, rd = read("REPORT.md"), read("README.md")
    bad = []
    # 小数第4位までの数値は実験値なので、README に出るなら本文にも同じ値があるはず
    for m in re.finditer(r"\d+\.\d{4}", rd):
        v = m.group()
        if v not in rep:
            i = rd.index(v)
            bad.append((v, rd[max(0, i - 34):i + 12].replace("\n", " ")))
    return bad


def check_files():
    """C: README の一覧と実体の突き合わせ。"""
    rd = read("README.md")
    bad = []
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        b = os.path.basename(f)
        if b not in rd:
            bad.append(("README に無い", b))
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*.csv"))):
        b = os.path.basename(f)
        if b not in rd:
            bad.append(("README に無い", f"results/{b}"))
    for m in re.finditer(r"`([a-z0-9_]+\.py)`", rd):
        if not os.path.exists(os.path.join(HERE, m.group(1))):
            bad.append(("実体が無い", m.group(1)))
    return bad


def check_named():
    """D: 本文が名指ししたスクリプト・CSV が実在するか。"""
    rep = read("REPORT.md")
    bad = []
    for m in re.finditer(r"`([a-z0-9_]+\.(?:py|csv|md))`", rep):
        n = m.group(1)
        if not (os.path.exists(os.path.join(HERE, n))
                or os.path.exists(os.path.join(HERE, "results", n))):
            bad.append(n)
    return bad


def main():
    n = 0
    print("A 画像数の一致")
    for what, detail in check_images():
        print(f"  ★ {what}: {detail}")
        n += 1
    if not check_images():
        print("  （実体と記述が一致）")

    print("B README と本文で共通する実験値")
    for v, ctx in check_shared_numbers():
        print(f"  ★ {v} が本文に無い: …{ctx}…")
        n += 1
    if not check_shared_numbers():
        print("  （食い違いなし）")

    print("C README のファイル一覧")
    for what, f in check_files():
        print(f"  ★ {what}: {f}")
        n += 1
    if not check_files():
        print("  （過不足なし）")

    print("D 本文が名指ししたファイル")
    for f in check_named():
        print(f"  ★ 実体が無い: {f}")
        n += 1
    if not check_named():
        print("  （すべて実在）")


    print(f"\n食い違い {n} 件")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
