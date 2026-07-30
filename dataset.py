"""入力画像の用意。著作権の問題が起きない2系統だけを使う。

(1) skimage.data 同梱画像 — scikit-image に同梱され再配布が許諾されているもの。
    各画像の出所は LICENSE_DATA.md に記載する。
(2) 合成画像 — 本レポート用に cv2 で生成。エッジ密度を独立変数として
    制御できるので、密度と破綻の因果を切り分けるのに使う。
"""
import cv2
import numpy as np

SIZE = 512


def _fit(img):
    """512x512 の BGR に揃える。"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def real_images():
    """skimage.data の同梱画像。エッジ密度が低〜高まで散るように選ぶ。"""
    from skimage import data
    items = [
        ("astronaut", data.astronaut, "a photo of an astronaut in a spacesuit"),
        ("camera", data.camera, "a photo of a photographer with a tripod"),
        ("coffee", data.coffee, "a photo of a cup of coffee on a table"),
        ("chelsea", data.chelsea, "a photo of a cat"),
        ("coins", data.coins, "a photo of scattered metal coins"),
        ("horse", data.horse, "a photo of a horse silhouette"),
        ("page", data.page, "a photo of a printed page of text"),
        ("rocket", data.rocket, "a photo of a rocket on a launch pad"),
        ("immunohistochemistry", data.immunohistochemistry,
         "a microscope photo of stained tissue"),
        ("cell", data.cell, "a microscope photo of a single cell"),
    ]
    out = []
    for name, fn, prompt in items:
        try:
            arr = np.asarray(fn())
            if arr.dtype == bool:
                arr = (arr * 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = cv2.normalize(arr.astype(np.float32), None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            out.append((name, _fit(arr), prompt))
        except Exception as e:  # データが無い環境でも止めない
            print(f"  skip {name}: {e}")
    return out


def holdout_images():
    """検証用に取り分けた skimage.data の同梱画像。

    ★skimage.data.cat は chelsea のエイリアス（同一配列）なので入れない。
    開発側に chelsea があるため、入れると held-out にならない。

    第6節の改善と、その適用条件（密度が目標を下回るときだけ適用する）は
    real_images() の 10 点＋合成 5 点から作った。同じデータで評価すれば
    見積もりは楽観側に寄る。そこで開発に一切使っていない 8 点を別に取り、
    規則をそのまま適用して再現するかを見る。
    密度は α=1.0 で 0.0005〜0.3130 と、開発側（0.0016〜0.1884）より広い。
    """
    from skimage import data
    items = [
        ("brick", data.brick, "a photo of a brick wall"),
        ("clock", data.clock, "a photo of a wall clock"),
        ("grass", data.grass, "a photo of grass"),
        ("gravel", data.gravel, "a photo of gravel"),
        ("moon", data.moon, "a photo of the moon"),
        ("retina", data.retina, "a microscope photo of a retina"),
        ("hubble_deep_field", data.hubble_deep_field,
         "a photo of distant galaxies"),
    ]
    out = []
    for name, fn, prompt in items:
        try:
            arr = np.asarray(fn())
            if arr.dtype == bool:
                arr = (arr * 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = cv2.normalize(arr.astype(np.float32), None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            out.append((name, _fit(arr), prompt))
        except Exception as e:
            print(f"  skip {name}: {e}")
    return out


def synthetic_images():
    """エッジ密度を段階的に変えた合成画像。密度が唯一の変化要因になるよう
    物体の意味内容（円と矩形）は揃え、反復の細かさだけを変える。"""
    out = []
    specs = [("sparse", 1), ("low", 2), ("mid", 4), ("dense", 8), ("veryd", 14)]
    for name, n in specs:
        img = np.full((SIZE, SIZE, 3), 190, np.uint8)
        step = SIZE // (n + 1)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                c = (j * step, i * step)
                r = max(4, step // 3)
                cv2.circle(img, c, r, (40, 40, 40), 2)
                cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r),
                              (90, 90, 90), 1)
        out.append((f"synth_{name}", img,
                    "a photo of ceramic plates and bowls on a table"))
    return out


def contrast_series(img, alphas=(0.15, 0.3, 0.5, 1.0)):
    """コントラストだけを落とした系列。構図・内容は完全に同一なので、
    Canny 閾値が固定であることの影響だけを取り出せる。"""
    return [(a, cv2.convertScaleAbs(img, alpha=a,
                                    beta=int(128 * (1 - a)))) for a in alphas]


if __name__ == "__main__":
    r = real_images()
    s = synthetic_images()
    print(f"real={len(r)} synthetic={len(s)}")
    from pipeline import canny, edge_density
    for name, img, _ in r + s:
        print(f"  {name:24s} edge_density={edge_density(canny(img)):.4f}")
