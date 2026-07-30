"""生成済み画像の健全性を一括検査する。黙って壊れていないかを毎回確認する用。"""
import glob, os, sys
import cv2, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
bad = []
files = sorted(glob.glob(os.path.join(R, "images", "*_gen.png")) +
               glob.glob(os.path.join(R, "images", "*_before.png")) +
               glob.glob(os.path.join(R, "images", "*_after.png")))
for f in files:
    im = cv2.imread(f)
    if im is None: bad.append((f, "読込失敗")); continue
    if im.std() < 1.0: bad.append((f, f"退化 std={im.std():.2f} mean={im.mean():.1f}"))
print(f"検査 {len(files)}枚 / 異常 {len(bad)}枚")
for f, why in bad[:15]: print(f"  ★{os.path.basename(f)}: {why}")
sys.exit(1 if bad else 0)
