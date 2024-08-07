## From https://ai.gopubby.com/crazy-challenge-run-llama-405b-on-a-8gb-vram-gpu-ab5a280a3889

!pip install airllm

from airllm import AutoModel

model = AutoModel.from_pretrained（
    "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")

input_text = ['What is the capital of United States?',]

input_tokens = model.tokenizer(input_text,
      return_tensors="pt", 
      return_attention_mask=False, 
      truncation=True, 
      max_length=128, 
      padding=False)

generation_output = model.generate(
      input_tokens['input_ids'].cuda(), 
      max_new_tokens=10,
      return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])
