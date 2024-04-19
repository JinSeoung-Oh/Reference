# From https://towardsdatascience.com/orpo-preference-optimization-without-the-supervised-fine-tuning-sft-step-60632ad0f450

pip install -q -U bitsandbytes
pip install --upgrade -q -U transformers
pip install -q -U peft
pip install -q -U accelerate
pip install -q -U datasets
pip install -q -U git+https://github.com/huggingface/trl.git

import torch, multiprocessing
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import ORPOTrainer, ORPOConfig
import os

# Check GPU and FlashAttention
major_version, minor_version = torch.cuda.get_device_capability()
if major_version >= 8:
  os.system("pip install flash-attn")
  torch_dtype = torch.bfloat16
  attn_implementation='flash_attention_2'
  print("Your GPU is compatible with FlashAttention and bfloat16.")
else:
  torch_dtype = torch.float16
  attn_implementation='eager'
  print("Your GPU is not compatible with FlashAttention and bfloat16.")


# Load data
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=["train_prefs","test_prefs"])

def process(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

dataset[0] = dataset[0].map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

dataset[1] = dataset[1].map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

# Load model
model_name = "mistralai/Mistral-7B-v0.1"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' #Necessary for FlashAttention compatibility

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, torch_dtype=torch_dtype, quantization_config=bnb_config, device_map={"": 0},  attn_implementation=attn_implementation
)
model = prepare_model_for_kbit_training(model)
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)


# Train
orpo_config = ORPOConfig(
    output_dir="./results/",
    evaluation_strategy="steps",
    do_eval=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    log_level="debug",
    logging_steps=20,
    learning_rate=8e-6,
    eval_steps=20,
    max_steps=100,
    save_steps=20,
    save_strategy='epoch',
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    beta=0.1, #beta is ORPO's lambda
    max_length=1024,
)

trainer = ORPOTrainer(
        model=model,
        train_dataset=dataset[0],
        eval_dataset=dataset[1],
        peft_config=peft_config,
        args=orpo_config,
        tokenizer=tokenizer,
)

trainer.train()




