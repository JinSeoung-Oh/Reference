## From https://medium.com/@shivansh.kaushik/finetuning-llama3-using-axolotl-1becd616fc12

### Axolotl Config
"""
base_model: meta-llama/Meta-Llama-3-8B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: gbharti/finance-alpaca
    type: alpaca
dataset_prepared_path:
val_set_size: 0
output_dir: ./outputs/qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 1
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 1
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|end_of_text|>"
"""

#After git clone
cd axolotl

CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess llama3_qlora.yml
accelerate launch -m axolotl.cli.train llama3_qlora.yml

accelerate launch -m axolotl.cli.inference llama3_qlora.yml \
    --lora_model_dir="./outputs/qlora-out" --gradio
