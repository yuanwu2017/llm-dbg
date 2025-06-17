import os
import sys
import torch
import time
import soundfile
import logging

logging.basicConfig(level=logging.INFO)

from datasets import load_from_disk
from transformers import pipeline, set_seed
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")

inference_context = [torch.inference_mode()]
PROMPT = "Hello, my dog is cool !"


def generate(generator, forward_params, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(run_steps + warm_up_steps):
            set_seed(43)
            generator.forward_time = 0
            pre = time.time()
            output = generator(PROMPT, forward_params=forward_params)
            pipeline_times.append((time.time() - pre) * 1000)
            forward_times.append(generator.forward_time * 1000)

    return output, pipeline_times, forward_times


if __name__ == "__main__":
    warm_up_steps = 0
    run_steps = 1
    model_id = "facebook/hf-seamless-m4t-large"
    device = "hpu"

    torch_dtype = torch.bfloat16
    dtype = torch.bfloat16
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))
    synthesiser = pipeline(
        "text-to-speech", model_id, device=device, torch_dtype=torch_dtype
    )

    embeddings_dataset = load_from_disk("./speech_vector")
    speaker_embedding = (
        torch.tensor(embeddings_dataset[0]["xvector"])
        .unsqueeze(0)
        .to(device)
        .to(torch_dtype)
    )
    # by default the dtype of speaker_embedding is FP32, if the model dtype is not FP32, we need to manually convert it
    if torch_dtype != torch.float32:
        speaker_embedding = speaker_embedding.to(torch_dtype)

    # You can replace this embedding with your own as well.
    forward_params = (
        {"speaker_embeddings": speaker_embedding} if "t5" in model_id else {}
    )
    forward_params["do_sample"] = False
    forward_params["tgt_lang"] = "eng"

    output, pipeline_times, forward_times = generate(
        synthesiser, forward_params, warm_up_steps, run_steps
    )

    logging.info(f"output = {output}")

    audio_path = "audio_" + model_id.replace("/", "_") + ".wav"
    audio = output["audio"] if len(output["audio"].shape) == 1 else output["audio"][0]
    soundfile.write(
        audio_path, audio, output["sampling_rate"], format="WAV", subtype="PCM_16"
    )
