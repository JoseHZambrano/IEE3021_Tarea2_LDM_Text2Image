from pathlib import Path
import time

import torch
from diffusers import AutoPipelineForText2Image


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pipeline(model_id="stabilityai/sdxl-turbo"):
    device = get_device()

    if device.type == "cuda":
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = pipe.to(device)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id)
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=False)
    return pipe, device


def generate_image(
    pipe,
    prompt,
    seed,
    output_path,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512,
    width=512,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    start = time.perf_counter()

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    elapsed = time.perf_counter() - start

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return {
        "prompt": prompt,
        "seed": seed,
        "image_path": str(output_path),
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "time_seconds": elapsed,
    }