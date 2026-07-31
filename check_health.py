"""生成済み画像の健全性を一括検査する。黙って壊れていないかを毎回確認する用。

条件地図（*_cond.png）は密度0なら全画素0が正しいので対象から外す。
exp12 の slicingあり（修正なし）は、第4.1節が壊れることを示すために
わざと壊れた状態で保存してあるので、退化していることを期待する側で数える。
"""
import glob, os, re, sys
import cv2, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
bad, expected_broken = [], []
files = sorted(f for f in glob.glob(os.path.join(R, "images", "*.png"))
               if not f.endswith("_cond.png"))
for f in files:
    im = cv2.imread(f)
    if im is None: bad.append((f, "読込失敗")); continue
    degraded = im.std() < 1.0
    # 「slicingあり」で「修正」を含まないものだけが、壊れていることを期待する条件
    is_expected = ("e12_slicingあり_" in os.path.basename(f))
    if is_expected:
        (expected_broken if degraded else bad).append(
            (f, "壊れているはずが正常" if not degraded else "想定どおり退化"))
    elif degraded:
        bad.append((f, f"退化 std={im.std():.2f} mean={im.mean():.1f}"))
print(f"検査 {len(files)}枚 / 異常 {len(bad)}枚"
      f" / 想定どおり壊れている {len(expected_broken)}枚（第4.1節の証拠）")
for f, why in bad[:15]: print(f"  ★{os.path.basename(f)}: {why}")
sys.exit(1 if bad else 0)
