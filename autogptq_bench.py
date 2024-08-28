from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import torch
import time
from optimum.habana.utils import get_hpu_memory_stats
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--use_hpu_graphs",
    action="store_true",
    help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
)

parser.add_argument(
    "--model_name_or_path",
    default="TheBloke/Llama-2-7b-Chat-GPTQ",
    type=str,
    required=True,
    help="Path to pre-trained model (on the HF Hub or locally).",
)


args = parser.parse_args()

adapt_transformers_to_gaudi()

#model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
# To use a different branch, change revision
# For example: revision="gptq-4bit-64g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main",
                                             torch_dtype = torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]

'''

print("\n\n*** Generate:")
model = model.eval().to("hpu")
input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to("hpu")

if args.use_hpu_graphs:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    model = wrap_in_hpu_graph(model)   

print(f"input_ids.device={input_ids.device}")

for i in range(3) :
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=128)
start = time.time()
for i in range(10) :
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=128)
    print(tokenizer.decode(output[0]))
end = time.time() - start
print(f"time = {end}s")
mem_stats = get_hpu_memory_stats("hpu:0")
print(f"memory_usage={mem_stats}")

# Inference can also be done using transformers' pipeline

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     repetition_penalty=1.1,
# )

#print(pipe(prompt_template)[0]['generated_text'])