import numpy as np
import torch
import argparse
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_image,
    load_numpy,
)
from PIL import Image as im

from diffusers import StableDiffusionPipeline,DDIMScheduler

enable_full_determinism()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="hpu",
    type=str,
    help="Path to pre-trained model",
)


args = parser.parse_args()

#from diffusers import AutoPipelineForInpainting
def get_inputs(device, output_type = "np", generator_device="cuda", dtype=torch.float, seed=0):
    if device == "hpu":
        device = "cpu"
        dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)
        
    latents = np.random.RandomState(seed).standard_normal((2, 4, 64, 64))
    latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
    inputs = {
        "prompt": "a photograph of an astronaut riding a horse",
        "latents": latents,
        "generator": generator,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": output_type,
        "num_images_per_prompt": 2,
    }
    
    return inputs


def test_stable_diffusion_ddim():
    sd_pipe = None
    if args.device == "hpu":
        init_kwargs = {
            "use_habana": True,
            "use_hpu_graphs": False,
            "gaudi_config": "Habana/stable-diffusion",
            "torch_dtype": torch.bfloat16,

        }
        from optimum.habana.diffusers import GaudiStableDiffusionPipeline,GaudiDDIMScheduler
        sd_pipe = GaudiStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, **init_kwargs)
        sd_pipe.scheduler = GaudiDDIMScheduler.from_config(sd_pipe.scheduler.config)
    else:
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(args.device)
    
    sd_pipe.set_progress_bar_config(disable=None)
    for output_type in ["np", "latent", "pil", "pt"] :
        inputs= get_inputs(args.device, output_type=output_type)
        image = sd_pipe(**inputs).images
        print(f"if output_type is {output_type}, the output image should be = {type(image)}")
        if isinstance(image, np.ndarray):
            assert image.shape == (2, 512, 512, 3)
        elif isinstance(image, torch.Tensor) :
            if output_type == "latent":
                assert image.shape == torch.Size([2, 4, 64, 64])
            else:
                assert image.shape == torch.Size([2, 3, 512, 512])
        elif isinstance(image, list):
            assert len(image) == 2
            

if __name__ == "__main__":
    test_stable_diffusion_ddim()
