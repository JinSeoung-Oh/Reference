## From https://towardsdatascience.com/multi-gpu-fine-tuning-for-llama-3-1-70b-with-fsdp-and-qlora-67a8a5b4f0d6

# Generate a configuration file for FSDP with Accelerate
!pip install --upgrade accelerate

# accelerate config
"""
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false                                                                                                                                                                 
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

from accelerate import Accelerator
accelerator = Accelerator()

fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
        
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.
    set_state_dict_type("FULL_STATE_DICT")

## When run : accelerate launch --config_file config_fsdp.yaml fine_tuning_FSDP_QLoRA.py

## https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#quantized-data-storage
import torch, os, multiprocessing
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from peft.utils.other import fsdp_auto_wrap_policy
from accelerate import Accelerator

accelerator = Accelerator()
set_seed(1234)
#use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
  os.system('pip install flash_attn')
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'
model_name = "meta-llama/Meta-Llama-3.1-70B"
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = 128004
tokenizer.padding_side = 'right'
ds = load_dataset("timdettmers/openassistant-guanaco")
#Add the EOS token
def process(row):
    row["text"] = row["text"]+"<|end_of_text|>"
    return row
ds = ds.map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
)
for name, param in model.named_parameters():
    # freeze base model's layers
    param.requires_grad = False
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)
output_dir = "./Llama3.1_70b_QLoRA/"

training_arguments = SFTConfig(
        output_dir=output_dir ,
        eval_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        log_level="debug",
        logging_steps=10,
        learning_rate=1e-4,
        bf16 = True,
        eval_steps=10,
        max_steps=50,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=512,
)
trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
)
fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
trainer.train()
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.
    fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(output_dir)





