"""黒画像の原因の初期切り分け（第4.1節に至る前の一回きりの診断）。

fp32 の VAE を使えば直るかを試したもので、結果は「直らない」。
原因の特定は exp8_slicing_cause.py 以降で行った。
実験の連鎖（run_all.sh）には含めていない。
"""
import torch, numpy as np, cv2, time
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
import dataset as ds, pipeline as P

cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=torch.float16)
p = StableDiffusionControlNetPipeline.from_pretrained(P.MODEL_SD, controlnet=cn,
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, algorithm_type="dpmsolver++")
p=p.to("mps"); p.enable_attention_slicing(); p.set_progress_bar_config(disable=True)

name,img,prompt = ds.real_images()[0]
cond = P.to_cond(P.canny(img))
g=torch.Generator(device="cpu").manual_seed(0)

t=time.time()
lat = p(prompt, image=cond, num_inference_steps=10, generator=g, output_type="latent").images
print(f"[latent] {time.time()-t:.0f}s shape={tuple(lat.shape)} dtype={lat.dtype}")
l=lat.float().cpu()
print(f"  NaN={torch.isnan(l).any().item()} Inf={torch.isinf(l).any().item()} "
      f"mean={l.mean():.4f} std={l.std():.4f} absmax={l.abs().max():.2f}")

# fp16 VAE で復号
with torch.no_grad():
    d16 = p.vae.decode(lat / p.vae.config.scaling_factor, return_dict=False)[0]
a16=d16.float().cpu().numpy()
print(f"[fp16 VAE] NaN={np.isnan(a16).any()} mean={np.nanmean(a16):.4f} std={np.nanstd(a16):.4f}")

# fp32 VAE で復号
p.vae.to(torch.float32)
with torch.no_grad():
    d32 = p.vae.decode(lat.float() / p.vae.config.scaling_factor, return_dict=False)[0]
a32=d32.float().cpu().numpy()
print(f"[fp32 VAE] NaN={np.isnan(a32).any()} mean={np.nanmean(a32):.4f} std={np.nanstd(a32):.4f}")
im=((a32[0].transpose(1,2,0)/2+0.5).clip(0,1)*255).astype(np.uint8)
cv2.imwrite("results/diag_fp32vae.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
print(f"  保存後: mean={im.mean():.1f} std={im.std():.1f}")
