## Have to check - https://towardsdatascience.com/the-ultimate-handbook-for-llm-quantization-88bb7cb0d9d7

"""
Quantization is a technique that converts the high precision weights of a neural network, 
typically represented in 32-bit floating point (FP32), 
into lower precision values like 16-bit floating point (FP16) or 8-bit integer (INT8).
This process reduces the model size and computational demands, enabling efficient deployment on devices with limited resources,
such as CPUs or mobile phones.

1. Example Calculation
   For a model with 400 million parameters stored in FP32
   - Memory footprint: 4×10^8params×4bytes=1.6GB
   After quantizing to INT8
   - Memory footprint: 4×10^8params×1byte=0.4GB
This reduction to one-fourth of the original size helps in decreasing memory usage and enhancing inference speed, 
though it may slightly compromise accuracy.

2. Quantization Methods
   - 1. Linear/Scale Quantization
        - Maps weight values from a high-precision range to a lower precision range linearly.
        - For example, the minimum value in the original range (Rmin) is mapped to the minimum value in the quantized range (Qmin),
          and the maximum value (Rmax) to the maximum quantized value (Qmax).
   - 2. Affine Quantization
        - Allows for more asymmetric range representation.
        - Uses parameters to represent ranges, applying a transformation equation to map values and then clipping out-of-range data.

3. Types of Quantization Techniques
    - 1. Post-Training Quantization (PTQ)
         - Applied after training, converting weights and activations from higher to lower precision.
         - Optimizes speed, memory, and power usage but may introduce quantization error, affecting model accuracy.
    - 2. Quantization-Aware Training (QAT)
         - Integrates quantization into the training process.
         - Maintains both full-precision and quantized versions during training, allowing the model to adapt to quantization
           effects while preserving gradient calculation precision, improving robustness to quantization.

4. Why Quantize?
   -1. Reduced Memory Footprint
       Quantization significantly reduces memory requirements, facilitating deployment on lower-end machines and edge devices 
       that support only integer storage.
   -2. Faster Inference
       Lower precision (integer) computations are faster than higher precision (float) computations. 
       Modern CPUs and GPUs with specialized instructions for lower-precision computations can further enhance inference speed.
   -3. Reduced Energy Consumption
       Hardware accelerators optimized for lower-precision computations can perform more operations per watt of energy,
       making quantized models more energy-efficient.
"""

## 1. LLM.int8() (Aug 2022)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_8bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

----------------------------------------------------------------
## 2. GPTQ (Oct 2022)
!pip install auto-gptq transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant_config)

----------------------------------------------------------------
## 3. QLoRA (May 2023)

!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    task_type="CAUSAL_LM"
)

from transformers import TrainingArguments

output_dir = "./models"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

----------------------------------------------------------------
## 4. AWQ (Jun 2023)

!!pip install autoawq transformers accelerate
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model_id = 'meta-llama/Llama-2-7b-hf'
quant_path = 'Llama2-7b-awq-4bit'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }

# Load model and tokenizer
model = AutoAWQForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

----------------------------------------------------------------
## 5. Quip# (Jul 2023)

git clone https://github.com/Cornell-RelaxML/quip-sharp.git
pip install -r requirements.txt
cd quiptools && python setup.py install && cd ../

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantizer import QuipQuantizer

model_name = "meta-llama/Llama-2-70b-hf"
quant_dir = "llama-70b_2bit_quip"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quant = QuipQuantizer(codebook="E8P12", dataset="redpajama")
quant.quantize_model(model, tokenizer, quant_dir)

----------------------------------------------------------------
## 6. GGUF (Aug 2023)

pip install ctransformers[cuda]
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Load LLM and Tokenizer
# Use `gpu_layers` to specify how many layers will be offloaded to the GPU.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-beta-GGUF",
    model_file="zephyr-7b-beta.Q4_K_M.gguf",
    model_type="mistral", gpu_layers=50, hf=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta", use_fast=True
)

# Create a pipeline
pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')
# Load LLM and Tokenizer
# Use `gpu_layers` to specify how many layers will be offloaded to the GPU.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-beta-GGUF",
    model_file="zephyr-7b-beta.Q4_K_M.gguf",
    model_type="mistral", gpu_layers=50, hf=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta", use_fast=True
)

# Create a pipeline
pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')

----------------------------------------------------------------
## 7. HQQ (Nov 2023)

from transformers import AutoModelForCausalLM, HqqConfig

# All linear layers will use the same quantization config
quant_config = HqqConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False, axis=1)

model_id = "meta-llama/Llama-2-7b-hf"

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config

----------------------------------------------------------------
## 8. AQLM (Feb 2024)
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf",
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf")
