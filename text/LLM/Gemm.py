# From https://medium.com/mlearning-ai/googles-gemma-fine-tuning-quantization-and-inference-on-your-computer-83066b25791b

# With vLLM
! pip install vllm
import time
from vllm import LLM, SamplingParams
prompts = [
    "The best recipe for pasta is"
]
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=150)
loading_start = time.time()
llm = LLM(model="google/gemma-7b")
print("--- Loading time: %s seconds ---" % (time.time() - loading_start))
generation_time = time.time()
outputs = llm.generate(prompts, sampling_params)
print("--- Generation time: %s seconds ---" % (time.time() - generation_time))
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
    print('------')

# With Transformers
! pip install --upgrade transformers accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(1234)  # For reproducibility
prompt = "The best recipe for pasta is"
checkpoint = "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="cuda")
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=150)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# Quantization of Gemma 7B
! pip install --upgrade transformers bitsandbytes accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig

set_seed(1234)  # For reproducibility
prompt = "The best recipe for pasta is"
checkpoint = "google/gemma-7b"
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="cuda")
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=150)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# Fine-tuning Gemma 7B with QLORA
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

model_name = "google/gemma-7b"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
tokenizer.padding_side = 'left'
ds = load_dataset("timdettmers/openassistant-guanaco")
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, device_map={"": 0}
)
model = prepare_model_for_kbit_training(model)
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
training_arguments = TrainingArguments(
        output_dir="./results_qlora",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        log_level="debug",
        save_steps=50,
        logging_steps=50,
        learning_rate=2e-5,
        eval_steps=50,
        max_steps=300,
        warmup_steps=30,
        lr_scheduler_type="linear",
)
trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
)
trainer.train()
