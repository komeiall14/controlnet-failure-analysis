"""REPORT.md から提出用 PDF を作る。

方針はメモリ reference_report_pdf_pipeline に従う。
- /usr/bin/python3（Apple 製）の reportlab を使う。Homebrew 側には入っていない
- 本文＝游明朝、見出し＝游ゴシック太字（Word 同梱 DFonts）
- wordWrap='CJK' ＋ 左揃え。TA_JUSTIFY は日本語で間延びする
- 文字は全て黒。A4、余白 20mm 前後
- 生成後に欠字ゼロとページ数を検証する

使い方: /usr/bin/python3 make_pdf.py
"""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.platypus import paraparser

# 下付き・上付きの既定の変位（フォントサイズの 0.5 倍）は本文の行間に対して深く、
# y_c の c が次行のインクに接近して別グリフのように見えていた。浅くする。
paraparser.subFraction = 0.22
paraparser.supFraction = 0.40

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "REPORT.md")
OUTNAME = "48266454_新井滉明_映像メディア学_レポート.pdf"
OUT = os.path.join(HERE, OUTNAME)
# 提出物3（生成AI利用ログ）。設問が別項目として挙げているので独立した1枚にする
LOG_SRC = os.path.join(HERE, "AI_USAGE_LOG.md")
LOG_OUT = os.path.join(HERE, "48266454_新井滉明_映像メディア学_生成AI利用ログ.pdf")

DF = "/Applications/Microsoft Word.app/Contents/Resources/DFonts"
pdfmetrics.registerFont(TTFont("Mincho", os.path.join(DF, "yumin.ttf")))
pdfmetrics.registerFont(TTFont("GothicB", os.path.join(DF, "YuGothB.ttc"),
                               subfontIndex=0))

# ※allowWidows/allowOrphans は wordWrap='CJK' と併用すると
#   reportlab が FragLine で落ちるので使わない。
BODY = ParagraphStyle("body", fontName="Mincho", fontSize=9.9, leading=13.6,
                      firstLineIndent=9.9, alignment=TA_LEFT,
                      wordWrap="CJK", textColor=colors.black,
                      spaceAfter=2)
NOIND = ParagraphStyle("noind", parent=BODY, firstLineIndent=0)
H1 = ParagraphStyle("h1", fontName="GothicB", fontSize=15, leading=20,
                    spaceBefore=2, spaceAfter=8, textColor=colors.black,
                    wordWrap="CJK")
H2 = ParagraphStyle("h2", fontName="GothicB", fontSize=11.6, leading=16,
                    spaceBefore=6, spaceAfter=2.5, textColor=colors.black,
                    wordWrap="CJK")
H3 = ParagraphStyle("h3", fontName="GothicB", fontSize=10.8, leading=15,
                    spaceBefore=5.5, spaceAfter=2.5, textColor=colors.black,
                    wordWrap="CJK")
META = ParagraphStyle("meta", parent=NOIND, fontSize=9.4, leading=13.4)
# 参考文献は本文より一段小さく組む（学術誌の慣例）
REF = ParagraphStyle("ref", parent=NOIND, fontSize=9.0, leading=12.2)
CAP = ParagraphStyle("cap", parent=NOIND, fontSize=8.8, leading=12,
                     textColor=colors.black)


def inline(s):
    """Markdown の強調とコードを reportlab のタグへ。数式の下付き上付きも変換する。"""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r'<font name="GothicB">\1</font>', s)
    s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", s)   # 斜体（誌名など）
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"_\{(.+?)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{(.+?)\}", r"<super>\1</super>", s)
    # 波括弧なしの下付き・上付き（y_c, Θ_z1, h_1, w_i など）。
    # 識別子直後の _英数字 を拾う。前後が空白や記号の場合は対象外にして誤変換を防ぐ。
    s = re.sub(r"(?<=[A-Za-zΘΩα-ω])_([A-Za-z0-9]{1,3})\b", r"<sub>\1</sub>", s)
    s = re.sub(r"(?<=[A-Za-z0-9)])\^([A-Za-z0-9]{1,3})\b", r"<super>\1</super>", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)
    return s


def build(src=None, out=None):
    src, out = src or SRC, out or OUT
    lines = open(src, encoding="utf-8").read().split("\n")
    story, tbl = [], []

    def flush_table():
        if not tbl:
            return
        rows = [[Paragraph(inline(c), CAP) for c in r] for r in tbl]
        w = (A4[0] - 40 * mm) / max(len(rows[0]), 1)
        t = Table(rows, colWidths=[w] * len(rows[0]))
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 1.8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ]))
        story.append(t)
        story.append(Spacer(1, 5))
        tbl.clear()

    in_code = False
    in_refs = False
    for ln in lines:
        s = ln.rstrip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if s.startswith("<!--"):
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            tbl.append(cells)
            continue
        flush_table()
        if not s:
            continue
        if s.startswith("---"):
            continue
        if in_code:
            story.append(Paragraph(inline(s).replace(" ", "&nbsp;"),
                                   ParagraphStyle("code", parent=NOIND,
                                                  fontSize=8.6, leading=11.6)))
            continue
        if s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), H3))
        elif s.startswith("## "):
            in_refs = s.startswith("## 参考文献")
            story.append(Paragraph(inline(s[3:]), H2))
        elif s.startswith("# "):
            story.append(Paragraph(inline(s[2:]), H1))
        elif s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.+?)\)", s)
            if m and os.path.exists(os.path.join(HERE, m.group(2))):
                p = os.path.join(HERE, m.group(2))
                from PIL import Image as PILImage
                iw, ih = PILImage.open(p).size
                # 縦長の図（散布図）は幅を詰める。横長の並び図は情報密度が高いので
                # 幅を保つ。どちらも本文幅を超えない範囲に収める。
                # 横長の並び図は情報密度が高いので幅を保つ。
                # 縦長の散布図は大きくしても情報が増えないので抑える。
                # 146mm は図2が第5節の途中のページに収まる上限。これより広いと
                # KeepTogether の塊が次ページへ送られ、前ページの下端が大きく空く。
                wmm = 146 if iw / ih > 2.0 else 138
                w = min(A4[0] - 36 * mm, wmm * mm)
                # 図とキャプションは離さない。ただし塊にすると前ページに
                # 収まらないときに丸ごと送られて下端が空くので、
                # 図の直前に改ページを許す余地を残す。
                from reportlab.platypus import KeepTogether
                block = [Image(p, width=w, height=w * ih / iw)]
                if m.group(1):
                    block.append(Paragraph(inline(m.group(1)), CAP))
                story.append(KeepTogether(block))
                story.append(Spacer(1, 4))
        elif re.match(r"^[-*] ", s):
            story.append(Paragraph("・" + inline(s[2:]), NOIND))
        elif re.match(r"^\d+\. ", s):
            story.append(Paragraph(inline(s), NOIND))
        elif s.startswith("    "):
            story.append(Paragraph(inline(s.strip()),
                                   ParagraphStyle("eq", parent=NOIND,
                                                  fontSize=9.6, leading=13)))
        else:
            head_block = len(story) < 8 and (
                "／" in s or s.startswith(("映像メディア学", "研究テーマ", "ソースコード")))
            story.append(Paragraph(inline(s),
                                   META if head_block else REF if in_refs else BODY))
    flush_table()

    doc = SimpleDocTemplate(out, pagesize=A4, topMargin=18 * mm,
                            bottomMargin=17 * mm, leftMargin=19 * mm,
                            rightMargin=19 * mm, title=OUTNAME[:-4])
    doc.build(story)
    return out


def orphan_lines(reader):
    """句読点だけが行頭へ送られた行を、描画位置から拾う。"""
    out = []
    for i, page in enumerate(reader.pages, 1):
        rows = {}

        def visit(text, cm, tm, font, size, rows=rows):
            t = text.strip()
            if t:
                rows.setdefault(round(tm[5]), []).append(t)

        page.extract_text(visitor_text=visit)
        for y, parts in rows.items():
            line = "".join(parts).strip()
            if line and len(line) <= 2 and all(c in "。、）」・" for c in line):
                out.append((i, line))
    return out


def verify(path):
    """欠字ゼロ・マークアップ漏れなし・ページ数を確認する。"""
    import pypdf
    r = pypdf.PdfReader(path)
    n = len(r.pages)
    txt = "\n".join(p.extract_text() or "" for p in r.pages)
    ok = True
    print(f"  ページ数: {n}  （設問の上限は4〜8ページ）")
    if not 4 <= n <= 8:
        print("  ★ページ数が範囲外"); ok = False
    for bad in ("<font", "<sub>", "<super>", "<i>", "&amp;", "&lt;", "**", "<!--"):
        if bad in txt:
            print(f"  ★マークアップ漏れ: {bad}"); ok = False
    # 作業中の申し送りを提出物へ混入させない。PDF だけでなく原稿も検査する。
    # REPORT.md は公開しているので、PDF に現れない箇所（HTML コメント等）も読める。
    src_all = open(SRC, encoding="utf-8").read()
    for memo in ("判断を仰ぎ", "TODO", "FIXME", "要確認", "後で直す", "★申し送り",
                 "後で書く", "仮置き", "要相談"):
        if memo in txt:
            print(f"  ★提出PDFに申し送りが混入: {memo}"); ok = False
        elif memo in src_all:
            print(f"  ★原稿に申し送りが残っている（公開リポジトリで読める）: {memo}"); ok = False
    # 「執筆の経過」を語る言い回しも本文からは落とす。下書きに何を書いて何を消したかは
    # 提出物の内容ではない。ただし第9節は課題が訂正の申告を求める場所なので除外する。
    # あわせて、本文で「測った」と書いたものを Limitations で「測っていない」と
    # 書く型の食い違いを、節をまたいで矛盾しやすい語の組で突き合わせる。
    sec4 = src_all.split("### 4.1")[-1].split("### 4.2")[0] if "### 4.1" in src_all else ""
    sec7 = src_all.split("## 7. Limitations")[-1].split("## 8.")[0] if "## 7. Limitations" in src_all else ""
    for got, notyet in (("中身も直接測った", "中身そのものは観測していない"),
                        ("原因は加算元の未初期化バッファ", "原因を計算グラフの中で特定したわけで"),
                        ("溢れの線は薄い", "溢れが生じている箇所")):
        if got in sec4 and notyet in sec7:
            print(f"  ★第4.1節と第7節が矛盾: 「{got}」と「{notyet}」"); ok = False

    body = src_all.split("## 9. 生成AI利用")[0]
    # 「これから何を書くか」の予告も落とす。事実だけを書けばよい。
    for memo in ("書きかけ", "当初の誤り", "書いていたが", "明示しておきたい",
                 "撤回した点", "後から気づき",
                 "しておきたい", "断っておく", "残しておく", "触れておく"):
        if memo in body:
            print(f"  ★本文に執筆の経過が残っている（申告は第9節に集約する）: {memo}")
            ok = False
    if "<!--" in src_all:
        print("  ★原稿に HTML コメントが残っている（公開リポジトリで読める）"); ok = False
    # 「。」の直後の半角スペース。編集中に付いた空きが PDF では字間の乱れに見える
    # 執筆の経過は本文だけでなくコード側にも残る。提出するのはリポジトリ全体なので、
    # 実際に配布されるファイル（git の管理下にあるもの）を一括で見る。
    # REPORT.md は上で個別に見ており、第9節は訂正の申告を求められている場所なので除く。
    # make_pdf.py 自身は下の検出語そのものを持つので除く。
    import subprocess
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=HERE, check=True,
                                 capture_output=True, text=True).stdout.split("\n")
    except Exception:
        tracked = []
    for rel in tracked:
        if (not rel.endswith((".py", ".sh", ".md"))
                or os.path.basename(rel) in ("REPORT.md", "make_pdf.py")):
            continue
        f = os.path.join(HERE, rel)
        if not os.path.exists(f):
            continue
        t = open(f, encoding="utf-8").read()
        for memo in ("書きかけ", "実際にやった", "実際に一度", "やらかし",
                     "TODO", "FIXME", "要相談", "判断を仰ぎ"):
            if memo in t:
                print(f"  ★{rel} に作業中の書き込み: {memo}"); ok = False

    if "。 " in src_all:
        i = src_all.index("。 ")
        print(f"  ★「。」の直後に半角スペース: …{src_all[max(0, i - 20):i + 12]}…")
        ok = False
    # 行頭に句読点だけが送られる（禁則が効かない）箇所。wordWrap="CJK" は
    # 行頭禁則を持たないので、折り返しの位置しだいで「。」だけの行ができる。
    for pg, line in orphan_lines(r):
        print(f"  ★{pg}ページ目に句読点だけの行: 「{line}」")
        ok = False
    face = pdfmetrics.getFont("Mincho").face
    # charToGlyph のキーは文字ではなくコードポイント。文字で引くと必ず None を
    # 返すので、全文字を欠字と誤判定する。
    #
    # 欠字は抽出後のテキストからは探せない。グリフの無い文字は抽出の時点で
    # すでに NUL に化けており、ord > 0x2000 の条件に掛からないためである。
    # 上付きマイナス(U+207B)がこの穴をすり抜け、3.4×10⁻⁵ が PDF 上では
    # 3.4×10□⁵ と指数の符号を落とした形で出ていた。原稿側を検査する。
    src = open(SRC, encoding="utf-8").read()
    missing = {c for c in src if ord(c) > 0x2000 and c not in "\n\r\t"
               and face.charToGlyph.get(ord(c)) is None}
    if missing:
        print(f"  ★欠字 {len(missing)}種: {''.join(sorted(missing))[:40]}"); ok = False
    elif "\x00" in txt:
        print("  ★PDF 内に NUL（描画されなかった文字）がある"); ok = False
    else:
        print("  欠字: なし")
    print(f"  抽出文字数: {len(txt)}")
    return ok


def verify_log(path):
    """提出物3（生成AI利用ログ）を検める。設問が求める3項目と欠字を見る。"""
    import pypdf
    src = open(LOG_SRC, encoding="utf-8").read()
    txt = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    ok = True
    for need in ("使用した生成AI", "主な用途", "自分で修正・判断した箇所"):
        if need not in txt:
            print(f"  ★設問が求める項目が無い: {need}"); ok = False
    for bad in ("<font", "<sub>", "<super>", "**", "<!--", "TODO", "FIXME", "要確認"):
        if bad in txt or bad in src:
            print(f"  ★マークアップ漏れ／申し送り: {bad}"); ok = False
    face = pdfmetrics.getFont("Mincho").face
    missing = {c for c in src if ord(c) > 0x2000 and c not in "\n\r\t"
               and face.charToGlyph.get(ord(c)) is None}
    if missing:
        print(f"  ★欠字 {len(missing)}種: {''.join(sorted(missing))}"); ok = False
    else:
        print("  欠字: なし")
    return ok


if __name__ == "__main__":
    p = build()
    print(f"生成: {p}")
    ok = verify(p)
    q = build(LOG_SRC, LOG_OUT)
    print(f"生成: {q}（提出物3）")
    ok = verify_log(q) and ok
    # 提出物は PDF だけではない。コードや README と食い違ったまま
    # 出さないよう、横断の点検もここで通す。
    for mod in ("audit_flow", "audit_consistency"):
        try:
            m = __import__(mod)
            if m.main():
                print(f"  ★{mod} に指摘がある（`python3 {mod}.py` で確認）")
                ok = False
        except Exception as e:
            print(f"  ★{mod} が実行できない: {e}")
            ok = False
    sys.exit(0 if ok else 1)
