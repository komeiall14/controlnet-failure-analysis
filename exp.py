"""ControlNet の失敗条件を実験的に切り出す。

exp1  条件地図の密度と構造整合の関係（固定閾値の縮退）
exp2  controlnet_conditioning_scale の入力依存感度
exp3  改善（密度適応 Canny ＋ 密度連動 scale）の前後比較

いずれも同一 seed・同一プロンプトでペアを組み、対応のある比較にする。
使い方:  python3 exp.py smoke | exp1 | exp2 | exp3
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

import dataset as ds
import pipeline as P

OUT = P.OUT
IMGDIR = os.path.join(OUT, "images")
os.makedirs(IMGDIR, exist_ok=True)
STEPS = 20   # DPMSolver++ 2M は20stepで十分収束する。1枚あたり約100秒


def log_rows(name, rows):
    if not rows:
        return
    p = os.path.join(OUT, f"{name}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {p} ({len(rows)} rows)")


def run_one(tag, img, prompt, seed, scale=1.0, lo=100, hi=200, save=True):
    """1枚生成して構造整合を測る。共通の測定経路はここだけに置く。"""
    e = P.canny(img, lo, hi)
    dens = P.edge_density(e)
    gen, dt = P.generate(prompt, P.to_cond(e), seed, scale=scale, steps=STEPS)
    m = P.edge_f1(e, gen)
    ch = P.chamfer(e, gen)
    if save:
        cv2.imwrite(os.path.join(IMGDIR, f"{tag}_gen.png"), gen)
        cv2.imwrite(os.path.join(IMGDIR, f"{tag}_cond.png"), e)
    return dict(tag=tag, seed=seed, scale=scale, lo=lo, hi=hi,
                cond_density=dens, gen_density=m["gen_density"],
                precision=m["precision"], recall=m["recall"], f1=m["f1"],
                chamfer=ch, sec=round(dt, 1))


# ------------------------------------------------------------------ 改善

def adaptive_canny(img, target=0.045, lo_ratio=0.5, iters=12):
    """目標エッジ密度に最も近づく高閾値を12回の二分探索で探す。

    固定閾値 (100,200) は学習時の条件分布を想定した値だが、入力の
    コントラストが変われば密度が桁で動く。密度の方を揃えれば、
    ControlNet が学習した条件分布に近い入力を与えられる、という狙い。
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    a, b = 20.0, 400.0
    best, best_e = None, None
    for _ in range(iters):
        hi = (a + b) / 2
        e = P.canny(g, int(hi * lo_ratio), int(hi))
        d = P.edge_density(e)
        if best is None or abs(d - target) < abs(best - target):
            best, best_e = d, e
        if d > target:
            a = hi
        else:
            b = hi
    return best_e, best


def density_aware_scale(dens, target=0.045, base=1.0, k=0.6, lo=0.4, hi=1.4):
    """密度が高いほど条件が実効的に強く効くため scale を下げる線形則。
    追加の生成コストはゼロ。"""
    s = base - k * (dens - target) / max(target, 1e-6)
    return float(np.clip(s, lo, hi))


# ------------------------------------------------------------------ 実験

def smoke():
    imgs = ds.synthetic_images()[:1] + ds.real_images()[:1]
    rows = []
    t0 = time.time()
    for name, img, prompt in imgs:
        rows.append(run_one(f"smoke_{name}", img, prompt, seed=0))
        print(f"  {name}: f1={rows[-1]['f1']:.3f} "
              f"dens={rows[-1]['cond_density']:.4f} {rows[-1]['sec']}s",
              flush=True)
    print(f"[smoke] {len(rows)}枚 / 合計{time.time()-t0:.0f}s")
    log_rows("smoke", rows)


def exp1():
    """固定閾値のもとで、条件地図の密度が構造整合をどう決めるか。"""
    rows = []
    for name, img, prompt in ds.synthetic_images() + ds.real_images():
        for a, dim in ds.contrast_series(img):
            tag = f"e1_{name}_a{a}"
            r = run_one(tag, dim, prompt, seed=0)
            r.update(source=name, alpha=a)
            rows.append(r)
            print(f"  {tag}: dens={r['cond_density']:.4f} f1={r['f1']:.3f} "
                  f"{r['sec']}s", flush=True)
            log_rows("exp1_density", rows)


def expvar():
    """条件内のシード分散を測る。

    exp1 は1条件1シードなので、密度と F1 の関係に見える揺れが本当の効果なのか
    生成のばらつきなのかを区別できない。密度帯を代表する5条件について
    5シードずつ回し、条件内の標準偏差を出しておく。これが無いと
    「密度が下がると壊れる」という主張が分散に埋もれていないと言えない。
    """
    rows = []
    pool = ds.synthetic_images() + ds.real_images()
    # 密度が低い順に並べ、帯を代表する5点を取る
    scored = sorted(((P.edge_density(P.canny(im)), n, im, pr)
                     for n, im, pr in pool), key=lambda x: x[0])
    picks = [scored[i] for i in
             (0, len(scored)//4, len(scored)//2, 3*len(scored)//4, len(scored)-1)]
    for dens, name, img, prompt in picks:
        for seed in range(5):
            tag = f"ev_{name}_s{seed}"
            r = run_one(tag, img, prompt, seed=seed)
            r.update(source=name)
            rows.append(r)
            print(f"  {tag}: dens={r['cond_density']:.5f} f1={r['f1']:.3f}",
                  flush=True)
            log_rows("expvar_seed", rows)


def exp2():
    """同じ入力に対し conditioning_scale を振り、最適値が入力で動くか。"""
    rows = []
    targets = ds.synthetic_images()[::2] + ds.real_images()[:4]
    for name, img, prompt in targets:
        for sc in (0.2, 0.4, 0.6, 0.8, 1.0, 1.3):
            tag = f"e2_{name}_s{sc}"
            r = run_one(tag, img, prompt, seed=0, scale=sc)
            r.update(source=name)
            rows.append(r)
            print(f"  {tag}: dens={r['cond_density']:.4f} f1={r['f1']:.3f}",
                  flush=True)
            log_rows("exp2_scale", rows)


def exp2b():
    """scale 掃引の延長。

    exp2 では 0.2〜1.3 を振ったが、7入力すべてが上限の 1.3 で最良となり、
    掃引範囲が狭すぎたことが判明した。構造整合だけを見ると単調増加に見えるので、
    反転点を探すために 1.3 の外側まで延ばす。あわせて CLIP でプロンプト追従を測り、
    「条件を強めるほど良い」が片方の指標だけを見た錯覚かどうかを判定する。
    """
    rows = []
    targets = ds.synthetic_images()[::2] + ds.real_images()[:4]
    for name, img, prompt in targets:
        for sc in (1.6, 2.0, 2.5, 3.0):
            tag = f"e2_{name}_s{sc}"
            r = run_one(tag, img, prompt, seed=0, scale=sc)
            r.update(source=name)
            rows.append(r)
            print(f"  {tag}: dens={r['cond_density']:.4f} f1={r['f1']:.3f}",
                  flush=True)
            log_rows("exp2b_scale_ext", rows)


def exp3():
    """改善前後の対応比較。同一 seed・同一プロンプトでペアを組む。"""
    rows = []
    seeds = (0, 1, 2)
    for name, img, prompt in ds.synthetic_images() + ds.real_images():
        for a, dim in ds.contrast_series(img, alphas=(0.3, 1.0)):
            e_base = P.canny(dim)
            d_base = P.edge_density(e_base)
            e_ad, d_ad = adaptive_canny(dim)
            sc = density_aware_scale(d_ad)
            for seed in seeds:
                # before: 固定閾値 + scale 1.0
                gb, _ = P.generate(prompt, P.to_cond(e_base), seed, 1.0, STEPS)
                mb = P.edge_f1(e_base, gb)
                # after: 密度適応 Canny + 密度連動 scale
                ga, _ = P.generate(prompt, P.to_cond(e_ad), seed, sc, STEPS)
                ma = P.edge_f1(e_ad, ga)
                rows.append(dict(source=name, alpha=a, seed=seed,
                                 dens_before=d_base, dens_after=d_ad,
                                 scale_after=sc,
                                 f1_before=mb["f1"], f1_after=ma["f1"],
                                 rec_before=mb["recall"],
                                 rec_after=ma["recall"],
                                 chamfer_before=P.chamfer(e_base, gb),
                                 chamfer_after=P.chamfer(e_ad, ga)))
                print(f"  {name} a={a} s{seed}: f1 {mb['f1']:.3f}"
                      f" -> {ma['f1']:.3f} (scale {sc:.2f})", flush=True)
                log_rows("exp3_improvement", rows)
                cv2.imwrite(os.path.join(IMGDIR, f"e3_{name}_a{a}_s{seed}_before.png"), gb)
                cv2.imwrite(os.path.join(IMGDIR, f"e3_{name}_a{a}_s{seed}_after.png"), ga)




def exp4():
    """失敗条件の追加1: 条件とプロンプトの意味的衝突。

    学習時にプロンプトの50%が空文字に置換されているため、モデルは条件地図から
    意味を読み取るよう仕向けられている。ならば条件が示す物体とプロンプトが要求する
    物体が食い違ったとき、両者は競合するはずである。同一の条件地図に対し、
    元カテゴリのプロンプトと別カテゴリのプロンプトを与えて構造整合を比較する。
    """
    rows = []
    conflict = ["a red sports car", "a wooden sailboat", "a bowl of ramen"]
    for name, img, prompt in ds.real_images()[:5]:
        e = P.canny(img)
        for kind, pr in [("match", prompt)] + [(f"conflict{i}", c)
                                               for i, c in enumerate(conflict)]:
            tag = f"e4_{name}_{kind}"
            gen, _ = P.generate(pr, P.to_cond(e), 0, 1.0, STEPS)
            m = P.edge_f1(e, gen)
            rows.append(dict(tag=tag, source=name, kind=kind, prompt=pr,
                             cond_density=P.edge_density(e), f1=m["f1"],
                             precision=m["precision"], recall=m["recall"],
                             chamfer=P.chamfer(e, gen)))
            cv2.imwrite(os.path.join(IMGDIR, f"{tag}_gen.png"), gen)
            print(f"  {tag}: f1={m['f1']:.3f}", flush=True)
            log_rows("exp4_conflict", rows)


def exp5():
    """失敗条件の追加2: attention slicing による NaN の発生条件を地図化する。

    第4.1節では slicing を有効にすると NaN が出ることを示したが、どの条件で
    出るのかは調べていない。「壊れる」から「これらの条件で壊れる」へ精度を上げる。
    拡散を4ステップしか回さないので1条件20秒程度で済む。
    """
    import gc
    import torch
    from diffusers import (ControlNetModel, DDIMScheduler,
                           StableDiffusionControlNetPipeline)
    name, img, prompt = ds.real_images()[0]
    cond = P.to_cond(P.canny(img))
    rows = []
    for dtype_name, dtype in (("fp16", torch.float16), ("fp32", torch.float32)):
        for slicing in ("なし", "auto", "max", 1, 4):
            cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=dtype)
            p = StableDiffusionControlNetPipeline.from_pretrained(
                P.MODEL_SD, controlnet=cn, torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False)
            p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
            p = p.to("mps"); p.set_progress_bar_config(disable=True)
            if slicing != "なし":
                p.enable_attention_slicing(slicing)
            g = torch.Generator(device="cpu").manual_seed(0)
            t0 = time.time()
            try:
                lat = p(prompt, image=cond, num_inference_steps=4, generator=g,
                        output_type="latent").images.float().cpu()
                nan = bool(torch.isnan(lat).any())
                amax = float(lat.abs().max()) if not nan else float("nan")
                err = ""
            except Exception as ex:
                nan, amax, err = True, float("nan"), type(ex).__name__
            sec = time.time() - t0
            rows.append(dict(dtype=dtype_name, slicing=str(slicing),
                             nan=nan, absmax=amax, error=err, sec=round(sec, 2)))
            print(f"  {dtype_name} slicing={slicing}: NaN={nan} absmax={amax:.2f} "
                  f"{sec:.1f}秒", flush=True)
            log_rows("exp5_slicing", rows)
            del p, cn; gc.collect(); torch.mps.empty_cache()


def exp6():
    """一般化の検証: 同じ現象が depth 条件でも起きるか。

    第4.2節と第5節の主張は Canny 条件でしか確かめていない。密度が下がると
    残差が縮むという説明が条件の種類に依存しないなら、depth 条件でも同じ関係が
    出るはずである。depth 地図は連続値なので「エッジ画素率」の代わりに
    条件地図の分散（情報量の代理）を横軸に取り、コントラストを落として掃引する。

    ControlNet 単体の順伝播だけを使うので1条件1秒未満で済み、拡散は回さない。
    """
    import gc

    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",
                                         torch_dtype=torch.float16)
    p = StableDiffusionControlNetPipeline.from_pretrained(
        P.MODEL_SD, controlnet=cn, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("mps")
    p.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def resid(gray, prompt):
        c = torch.from_numpy(np.stack([gray] * 3, 0)[None].astype(np.float32) / 255.
                             ).to("mps", torch.float16)
        emb, _ = p.encode_prompt(prompt, "mps", 1, False)
        g = torch.Generator(device="cpu").manual_seed(0)
        lat = torch.randn(1, 4, 64, 64, generator=g).to("mps", torch.float16)
        down, mid = p.controlnet(lat, torch.tensor([500], device="mps"),
                                 encoder_hidden_states=emb,
                                 controlnet_cond=c, return_dict=False)
        per = [float(d.float().pow(2).mean().sqrt()) for d in down]
        m = float(mid.float().pow(2).mean().sqrt())
        return dict(mid_rms=m, down_rms_mean=float(np.mean(per)),
                    total_rms=float(np.sqrt(np.mean([x ** 2 for x in per + [m]]))))

    rows = []
    for name, img, prompt in ds.real_images():
        g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for a in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            # a=0 は完全に平坦な地図（情報ゼロ）。Canny の「エッジ画素率0」に対応する
            gg = cv2.convertScaleAbs(g0, alpha=a, beta=int(128 * (1 - a)))
            r = resid(gg, prompt)
            r.update(source=name, alpha=a, cond_std=float(gg.std()),
                     cond_range=int(gg.max()) - int(gg.min()))
            rows.append(r)
            print(f"  depth {name:22s} a={a:<4} std={gg.std():6.2f} "
                  f"total_rms={r['total_rms']:.4f}", flush=True)
            log_rows("exp6_depth_generality", rows)
    del p, cn; gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    dict(smoke=smoke, exp1=exp1, expvar=expvar, exp2=exp2, exp2b=exp2b,
         exp3=exp3, exp4=exp4, exp5=exp5, exp6=exp6)[sys.argv[1]]()
