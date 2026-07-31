"""ControlNet (ICCV 2023) failure-case analysis — 共通基盤.

M1 / MPS / fp16 で SD1.5 + sd-controlnet-canny を回し、
入力条件マップと生成結果の構造整合を定量化する。
"""
import os
import time

import cv2
import numpy as np

# torch と PIL は生成にしか要らないので、先頭では読み込まない。
# canny / edge_density / edge_f1 / chamfer は OpenCV と NumPy だけで完結し、
# GPU の無い環境でも本文の数値を再現できる（README の検証手順がこの経路を使う）。
# torch と PIL は get_pipe / generate / to_cond の中で読む。

MODEL_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_CN = "lllyasviel/sd-controlnet-canny"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

_pipe = None


def get_pipe():
    """パイプラインは1度だけ構築する（ロードに40秒前後かかるため）。"""
    import torch
    global _pipe
    if _pipe is not None:
        return _pipe
    from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetPipeline)
    cn = ControlNetModel.from_pretrained(MODEL_CN, torch_dtype=torch.float16)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_SD, controlnet=cn, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    # UniPC は corrector 内で torch.linalg.solve を呼ぶが MPS 未実装のため使えない。
    # DPMSolver++ 2M は閉形式更新のみで linalg を使わず、20-25step で同等品質。
    p.scheduler = DPMSolverMultistepScheduler.from_config(
        p.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    p = p.to("mps")
    # enable_attention_slicing() は使わない。
    # HuggingFace の MPS 最適化ドキュメントは 64GB 未満の RAM で有効化を推奨しているが、
    # 本環境 (M1 / torch 2.2.2 / fp16) では slicing 有効時に UNet の順伝播で NaN が発生し、
    # 例外を出さないまま黒画像が返る。潜在変数の段階で NaN を確認済み (diag2.py)。
    # 無効にすると同一 seed で正常に生成される。
    p.set_progress_bar_config(disable=True)
    _pipe = p
    return p


# ---------------------------------------------------------------- 条件マップ

def canny(img_bgr, lo=100, hi=200):
    """論文および公式実装のデフォルト閾値は (100, 200)。"""
    return cv2.Canny(img_bgr, lo, hi)


def edge_density(edges):
    """条件マップのエッジ画素率。失敗条件を並べる横軸に使う。"""
    return float((edges > 0).mean())


def to_cond(edges):
    from PIL import Image
    return Image.fromarray(np.stack([edges] * 3, -1))


# ---------------------------------------------------------------- 評価指標

def edge_f1(cond_edges, gen_bgr, lo=100, hi=200, tol=2):
    """生成画像から同じ設定でエッジを再抽出し、入力条件との F1 を測る。

    tol ピクセルの許容を入れるのは、拡散生成では輪郭が数px揺れるため。
    許容なしだと構造が保たれていても F1 がほぼ0になり、指標として機能しない。
    """
    gen_edges = canny(gen_bgr, lo, hi)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    cond_d = cv2.dilate((cond_edges > 0).astype(np.uint8), k)
    gen_d = cv2.dilate((gen_edges > 0).astype(np.uint8), k)
    c = (cond_edges > 0).astype(np.uint8)
    g = (gen_edges > 0).astype(np.uint8)
    tp_p = float((g & cond_d).sum())          # 生成側エッジのうち条件に裏付けがある
    tp_r = float((c & gen_d).sum())           # 条件側エッジのうち生成に現れた
    prec = tp_p / max(g.sum(), 1)
    rec = tp_r / max(c.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(precision=prec, recall=rec, f1=f1,
                gen_density=edge_density(gen_edges))


def chamfer(cond_edges, gen_bgr, lo=100, hi=200):
    """条件エッジと生成エッジの双方向 Chamfer 距離（px）。F1 を補う連続量。"""
    gen_edges = canny(gen_bgr, lo, hi)
    if (cond_edges > 0).sum() == 0 or (gen_edges > 0).sum() == 0:
        return float("nan")
    dt_c = cv2.distanceTransform(255 - (cond_edges > 0).astype(np.uint8) * 255,
                                 cv2.DIST_L2, 3)
    dt_g = cv2.distanceTransform(255 - (gen_edges > 0).astype(np.uint8) * 255,
                                 cv2.DIST_L2, 3)
    a = dt_c[gen_edges > 0].mean()
    b = dt_g[cond_edges > 0].mean()
    return float((a + b) / 2)


# ---------------------------------------------------------------- 生成

def generate(prompt, cond_img, seed, scale=1.0, steps=25, negative=None):
    import torch
    p = get_pipe()
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = time.time()
    out = p(prompt, image=cond_img, num_inference_steps=steps,
            controlnet_conditioning_scale=float(scale),
            negative_prompt=negative, generator=g)
    dt = time.time() - t
    rgb = np.array(out.images[0])
    # 出力の健全性を毎回検査する。本環境では attention slicing 有効時に
    # 例外を出さないまま NaN が伝播し、全画素 0 の画像が返る（diag2.py）。
    # 静かに壊れる失敗は下流の指標を全て無意味にするので、生成の直後に落とす。
    if not np.isfinite(rgb).all():
        raise RuntimeError(f"生成結果に NaN/Inf: prompt={prompt[:40]!r} seed={seed}")
    if float(rgb.std()) < 1.0:
        raise RuntimeError(
            f"生成結果が退化（std={rgb.std():.3f} mean={rgb.mean():.1f}）: "
            f"黒画像・単色の疑い prompt={prompt[:40]!r} seed={seed}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dt
