from transformers import pipeline, AutoTokenizer
import torch

#model_id = "unsloth/Llama-4-Scout-17B-16E"
#model_id = "meta-llama/Llama-3.1-70B-Instruct"
model_id = "meta-llama/Llama-4-Scout-17B-16E"
#model_id = "meta-llama/Llama-2-70b-hf"

#messages = [
#    {"role": "user", "content": "what is the recipe of mayonnaise?"},
#]

messages="What is Deep Learning?"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    print(f"tokenizer.pad_token is None, setting pad_token to eos_token")
    tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 pad_token
tokenizer.pad_token_id = 0  # 设置 pad_token_id
tokenizer.padding_side = "left"   
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer_kwargs = {
    "padding": "max_length",  # 填充到 max_length
    "max_length": 128,        # 固定长度 128
    "truncation": True,       # 超长截断
    "pad_token_id": 0,  # 指定 pad_token
}
output = pipe(messages, do_sample=False, max_new_tokens=3,  **tokenizer_kwargs)
#print(output[0]["generated_text"][-1]["content"])
print(output)
