## From https://towardsdatascience.com/function-calling-fine-tuning-llama-3-on-xlam-f9b490d4f063
## https://huggingface.co/kaitchup/Meta-Llama-3-8B-xLAM-Adapter

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a calculator."},
    {"role": "user", "content": "Give me the square root of 3342398"},
]
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

if tool == "least_common_multiple":
    check = check_args_least_common_multiple(args)
    if check:
        return least_common_multiple(args["a"], args["b"])


ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
#Add the EOS token
def process(row):
    row["query"] = "<user>"+row["query"]+"</user>\n\n"
    tools = []
    for t in json.loads(row["tools"]):
      tools.append(str(t))
    answers = []
    for a in json.loads(row["answers"]):
      answers.append(str(a))
    row["tools"] = "<tools>"+"\n".join(tools)+"</tools>\n\n"
    row["answers"] = "<calls>"+"\n".join(answers)+"</calls>"
    row["text"] = row["query"]+row["tools"]+row["answers"]+tokenizer.eos_token
    return row
ds = ds.map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)


import torch, os, multiprocessing, json
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

#use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
  os.system('pip install flash_attn')
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'

model_name = "meta-llama/Meta-Llama-3-8B"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = "<|eot_id|>"
tokenizer.pad_token_id = 128009
tokenizer.padding_side = 'left'

def QLoRA(ds):
  bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=compute_dtype,
          bnb_4bit_use_double_quant=True,
  )
  model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map={"": 0}, attn_implementation=attn_implementation
  )
model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':True})
  #Configure the pad token in the model
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
  peft_config = LoraConfig(
          lora_alpha=16,
          lora_dropout=0.05,
          r=16,
          bias="none",
          task_type="CAUSAL_LM",
          target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
  )
  training_arguments = SFTConfig(
          output_dir="./Llama3_8b_xLAM",
          optim="adamw_8bit",
          per_device_train_batch_size=8,
          gradient_accumulation_steps=4,
          log_level="debug",
          save_steps=250,
          logging_steps=10,
          learning_rate=1e-4,
          fp16 = not torch.cuda.is_bf16_supported(),
          bf16 = torch.cuda.is_bf16_supported(),
          max_steps=1000,
          warmup_ratio=0.1,
          lr_scheduler_type="linear",
          dataset_text_field="text",
          max_seq_length=512,
  )
  trainer = SFTTrainer(
          model=model,
          train_dataset=ds,
          peft_config=peft_config,
          tokenizer=tokenizer,
          args=training_arguments,
  )
  trainer.train()

QLoRA(ds)

import torch,os
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

#use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
  os.system('pip install flash_attn')
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
adapter= "./Llama3_8b_xLAM/checkpoint-1000"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
print(f"Starting to load the model {model_name} into memory")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    device_map={"": 0},
    attn_implementation=attn_implementation,
)
print(model)
model = PeftModel.from_pretrained(model, adapter)

prompt = "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, do_sample=False, temperature=0.0, max_new_tokens=150)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

