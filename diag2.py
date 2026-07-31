"""黒画像がどの段階で生まれるかを切り分ける（第4.1節の前段）。

VAE で復号する前の潜在変数を取り出して NaN の有無を見る。
fp32 の VAE で復号し直しても消えないので、VAE の飽和ではなく
UNet の順伝播が壊れていることが分かる。attention slicing の有無でも比べる。
一回きりの診断なので run_all.sh には含めていない。
"""
import torch, time
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
import dataset as ds, pipeline as P

name,img,prompt = ds.real_images()[0]
cond = P.to_cond(P.canny(img))

def trial(label, dtype, slicing, steps=4):
    cn = ControlNetModel.from_pretrained(P.MODEL_CN, torch_dtype=dtype)
    p = StableDiffusionControlNetPipeline.from_pretrained(P.MODEL_SD, controlnet=cn,
            torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    p=p.to("mps"); p.set_progress_bar_config(disable=True)
    if slicing: p.enable_attention_slicing()
    g=torch.Generator(device="cpu").manual_seed(0)
    t=time.time()
    try:
        lat = p(prompt, image=cond, num_inference_steps=steps, generator=g,
                output_type="latent").images.float().cpu()
        nan = torch.isnan(lat).any().item()
        print(f"{label:34s} {time.time()-t:5.0f}s NaN={nan} "
              f"absmax={lat.abs().max():.2f}" if not nan else
              f"{label:34s} {time.time()-t:5.0f}s ★NaN発生", flush=True)
    except Exception as e:
        print(f"{label:34s} ★例外 {type(e).__name__}: {str(e)[:70]}", flush=True)
    del p, cn
    import gc; gc.collect(); torch.mps.empty_cache()

trial("fp16 + slicing (現行)", torch.float16, True)
trial("fp16 + slicingなし",    torch.float16, False)
trial("fp32 + slicingなし",    torch.float32, False)
