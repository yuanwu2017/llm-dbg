import numpy as np
import argparse
import torch
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_image,
)
from PIL import Image as im
from diffusers.utils import load_image
import time 
enable_full_determinism()
#Initialize inpaint parameters
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="hpu",
        type=str,
        help="Path to pre-trained model",
    )


    args = parser.parse_args()
    prompts = [
        "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k",
        "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
    ]
    num_images_per_prompt = 10
    num_inference_steps = 10
    model_name = ["runwayml/stable-diffusion-inpainting","diffusers/stable-diffusion-xl-1.0-inpainting-0.1"]
    
   
    init_kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": "Habana/stable-diffusion",
        "torch_dtype": torch.bfloat16,
    }
    
    if args.device == "cuda":
        init_kwargs = {
            "torch_dtype": torch.float16,
        }       
    if args.device == "hpu":    
        from optimum.habana.diffusers import AutoPipelineForInpainting
        from optimum.habana.utils import set_seed
    else:
        from diffusers import AutoPipelineForInpainting
    for model_name in model_name: 
        sdi_pipe = AutoPipelineForInpainting.from_pretrained(model_name, **init_kwargs)
        if args.device == "cuda":
            sdi_pipe.to("cuda")
        torch.manual_seed(0)
        if args.device =="hpu":
            set_seed(0)
        #warmup
        warmup = 2
        run_num = 5
        for i in range(warmup): 
            outputs = sdi_pipe(
                prompt=prompts,
                image=init_image,
                mask_image=mask_image,
                num_images_per_prompt=num_images_per_prompt,
                throughput_warmup_steps=3,
                num_inference_steps = num_inference_steps,
                batch_size=4
            )
        start = time.time()    
        for i in range(run_num):
            outputs = sdi_pipe(
                prompt=prompts,
                image=init_image,
                mask_image=mask_image,
                num_images_per_prompt=num_images_per_prompt,
                throughput_warmup_steps=0,
                num_inference_steps = num_inference_steps,
                batch_size=4
            )
        end = time.time()
        samples = num_images_per_prompt * len(prompts) * run_num
        runtime = end - start
        print(f"model_name = {model_name}, num_inference_steps = {num_inference_steps}, runtime = {runtime}, samples_per_second = {samples/runtime}")

if __name__ == "__main__":
    main()
