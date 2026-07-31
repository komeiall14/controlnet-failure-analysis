"""本文の論理のつながりを機械的に点検する。

数値は audit_numbers.py が照合する。こちらが見るのは文どうしのつながりで、
本文を削ったときに残りやすい破れ——削った内容を前提にした文、回収されない
予告、受け手を失った指示語——を対象にする。通読では見落とすので、
判定が決まる範囲だけでも機械にかける。

見るのは4つ。

  A 予告の回収   「第N節で〜する」と書いた先に、その語が実際にあるか
  B 用語の初出   主要な用語が、説明される前に使われていないか
  C 指示語       段落が「これ」「その」等で始まるとき、直前の段落があるか
  D 記号の定義   数式に出る記号が本文で説明されているか

いずれも「疑わしい箇所」を挙げるだけで、正しさの証明はしない。
ゼロ件が目標ではなく、挙がったものを人が見て判断する。

    python3 audit_flow.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "REPORT.md")

# 説明されるべき主要な用語と、その説明がある節
TERMS = {
    "zero convolution": "2",
    "trainable copy": "2",
    "controlnet_conditioning_scale": "2",
    "guess_mode": "2",
    "edge F1": "4.2",
    "構造整合": "4.2",
    "CLAHE": "6",
    "CLIP": "8",
    "Chamfer": "4.2",
}


def sections(txt):
    """節番号 -> その節の本文 の辞書を返す。"""
    out, cur, buf = {}, None, []
    for line in txt.split("\n"):
        m = re.match(r"^#{2,3} (\d+(?:\.\d+)?)[.．]?\s*(.*)", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), [m.group(2)]
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def paragraphs(txt):
    body = txt.split("## 参考文献")[0]
    return [p for p in body.split("\n")
            if p.strip() and not p.startswith(("#", "|", "!", "    "))]


def check_promises(txt, secs):
    """A: 「第N節で〜する」という前方への予告が回収されているか。

    後方参照（「第N節で述べた」「第N節で見た」）は既出の確認なので対象外。
    予告の文から内容語（名詞・術語）だけを拾い、その節に現れるかを見る。
    """
    STOP = set("する 見る 述べる 示す 扱う 測る 得る 触れる 挙げる 使う 書く "
               "この その ある いる なる ため こと もの よう とおり ように "
               "および ならびに つまり しかし ただし なお また さらに".split())
    bad = []
    for m in re.finditer(r"第 ?(\d+(?:\.\d+)?) ?節(?:で|では)([^。]{2,50})。", txt):
        n, what = m.group(1), m.group(2)
        # 後方参照は対象外
        if re.search(r"(述べた|見た|示した|測った|扱った|触れた|挙げた|得た|書いた"
                     r"|述べている|示している|とおり|ように)", what):
            continue
        # 前方への予告だけを見る
        if not re.search(r"(する|測る|見る|示す|扱う|評価する|延ばす|比べる)$",
                         what.strip()):
            continue
        if n not in secs:
            bad.append((n, what.strip(), "節が存在しない"))
            continue
        # 内容語（3字以上の漢字・カタカナ語・英数字）を拾う
        words = [w for w in re.findall(r"[一-龠ァ-ヶA-Za-z_0-9]{3,}", what)
                 if w not in STOP]
        if not words:
            continue
        if not any(w in secs[n] for w in words):
            bad.append((n, what.strip()[:34], f"『{words[0]}』が第{n}節に無い"))
    return bad


def check_terms(txt, secs):
    """B: 用語が、説明も付けずに説明節より前で使われていないか。

    初出の文にその場の言い換え（「〜という」「〜で測る」「（…）」など）が
    付いていれば、読者は追えるので問題にしない。
    """
    order = sorted(secs, key=lambda x: [int(v) for v in x.split(".")])
    pos = {s: i for i, s in enumerate(order)}
    bad = []
    for term, home in TERMS.items():
        if home not in pos:
            continue
        for s in order:
            if term not in secs[s]:
                continue
            if pos[s] >= pos[home]:
                break
            # 初出の文を取り出し、その場に説明が添えてあるかを見る
            i = txt.index(term)
            st, en = txt.rfind("。", 0, i) + 1, txt.find("。", i) + 1
            sent = txt[st:en]
            glossed = (re.search(r"という|とは|すなわち|＝|定義は|\(|（", sent)
                       or re.search(r"第 ?\d+(\.\d+)? ?節", sent))
            if not glossed:
                bad.append((term, s, home))
            break
    return bad


def check_demonstratives(txt):
    """C: 指示語で始まる段落に、受けるべき直前の段落があるか。"""
    heads = ("これ", "それ", "この", "その", "こう", "そう", "同様", "前述", "上述", "後者", "前者")
    body = txt.split("## 参考文献")[0].split("\n")
    bad, prev_is_para = [], False
    for line in body:
        s = line.strip()
        if not s:
            continue
        is_para = not s.startswith(("#", "|", "!", "    ", "---"))
        if is_para:
            plain = re.sub(r"[*`]", "", s)
            if plain.startswith(heads) and not prev_is_para:
                bad.append(plain[:44])
        prev_is_para = is_para
    return bad


def check_symbols(txt):
    """D: 数式の記号が本文で説明されているか。"""
    # コード片（torch. などを含む行）は数式ではないので除く
    eq = [l.strip() for l in txt.split("\n")
          if l.startswith("    ") and "=" in l and "torch" not in l]
    bad = []
    for line in eq:
        for sym in set(re.findall(r"(?<![A-Za-z_])([A-ZΘΩ])(?![A-Za-z_])", line)):
            # その記号が「ここで A は」「A が」の形で説明されているか
            if not re.search(rf"{sym} (?:は|が|を)", txt):
                bad.append((sym, line[:48]))
    return bad


def main():
    txt = open(SRC, encoding="utf-8").read()
    secs = sections(txt)
    n = 0

    print("A 予告の回収")
    for sec, what, why in check_promises(txt, secs):
        print(f"  ★ 第{sec}節への予告『{what}』… {why}")
        n += 1
    print("  （疑わしいものなし）" if not check_promises(txt, secs) else "")

    print("B 用語の初出")
    for term, used, home in check_terms(txt, secs):
        print(f"  ★ 『{term}』が第{used}節で使われているが、説明は第{home}節")
        n += 1
    if not check_terms(txt, secs):
        print("  （説明より前に使われている用語なし）")

    print("C 指示語で始まる段落")
    for s in check_demonstratives(txt):
        print(f"  ★ 受ける段落が直前にない: {s}…")
        n += 1
    if not check_demonstratives(txt):
        print("  （宙に浮いた指示語なし）")

    print("D 数式の記号")
    for sym, line in check_symbols(txt):
        print(f"  ★ 記号 {sym} の説明が見当たらない: {line}…")
        n += 1
    if not check_symbols(txt):
        print("  （説明のない記号なし）")

    print(f"\n疑わしい箇所 {n} 件")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
