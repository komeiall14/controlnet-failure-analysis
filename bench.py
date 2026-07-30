import time, torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

t0=time.time()
cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn,
    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps"); pipe.enable_attention_slicing()
print(f"[load] {time.time()-t0:.1f}s", flush=True)

img = np.zeros((512,512,3), np.uint8)
cv2.circle(img,(256,256),150,(255,255,255),-1); cv2.rectangle(img,(120,120),(240,240),(0,0,0),-1)
edges = cv2.Canny(img,100,200)
cond = Image.fromarray(np.stack([edges]*3,-1))

for steps in (20,25):
    t=time.time()
    out = pipe("a photo of a red apple on a wooden table", image=cond,
               num_inference_steps=steps, generator=torch.manual_seed(0))
    dt=time.time()-t
    out.images[0].save(f"bench_{steps}.png")
    print(f"[gen] steps={steps}: {dt:.1f}s ({dt/steps:.2f}s/step)", flush=True)
