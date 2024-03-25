"""
From https://medium.com/@geronimo7/llm-training-on-consumer-gpus-with-galore-d25075143cfb
See : https://github.com/geronimi73/3090_shorts/blob/main/nb_galore_llama2-7b.ipynb

GaLore reduces VRAM requirements not by decreasing the number of parameters directly but by optimizing how these parameters are trained.

GaLore focuses on two primary strategies:

1. Gradient Low-Rank Projection
   GaLore shifts away from handling the full, high-dimensional gradients of weight matrices. 
   Instead, it projects these gradients onto a low-rank space, significantly reducing the computational load 
   while retaining essential information for training.
2. Per-Layer Weight Updates
   Unlike the conventional method where an optimizer updates all layers simultaneously after backpropagation, 
   GaLore implements updates on a per-layer basis during backpropagation. 
   This approach further reduces the memory footprint throughout the training process.

Just like LoRA, GaLore allows us to finetune a 7B model on a consumer GPU with 24 GB VRAM. 
The performance of the resulting model is comparable to a full parameter finetune and appears to outperform LoRA.
"""

! pip install galore-torch
! pip install datasets==2.18.0
! pip install transformers==4.39.1
! pip install trl==0.8.1
! pip install accelerate==0.28.0
! pip install torch==2.2.1

#### Dummy classes for scheduler and optimizer ####
from typing import Optional
import torch

# Approach taken from Hugging Face transformers https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
class LayerWiseDummyOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer_dict=None, *args, **kwargs):
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": 1e-03})

    def zero_grad(self, set_to_none: bool = True) -> None: 
      pass

    def step(self, closure=None) -> Optional[float]: 
      pass

class LayerWiseDummyScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, *args, **kwargs):
        optimizer = LayerWiseDummyOptimizer()
        last_epoch = -1
        verbose = False
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self): 
      return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self): 
      return self.base_lrs


#### Load GaLore optimizer ####
from transformers import get_constant_schedule
from functools import partial
import torch.nn
import bitsandbytes as bnb

from galore_torch import GaLoreAdamW8bit
        
def load_galore_optimizer(model, lr, galore_config):    
    # function to hook optimizer and scheduler to a given parameter 
    def optimizer_hook(p, optimizer, scheduler):
        if p.grad is not None: 
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    # Parameters to optimize with Galore
    galore_params = [
        (module.weight, module_name) for module_name, module in model.named_modules() 
        if isinstance(module, nn.Linear) and any(target_key in module_name for target_key in galore_config["target_modules_list"])
    ] 
    id_galore_params = {id(p) for p, _ in galore_params}
    
    # Hook Galore optim to all target params, Adam8bit to all others
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in id_galore_params:
                optimizer = GaLoreAdamW8bit([dict(params=[p], **galore_config)], lr=lr)
            else:
                optimizer = bnb.optim.Adam8bit([p], lr = lr)
            scheduler = get_constant_schedule(optimizer)
            
            p.register_post_accumulate_grad_hook(partial(optimizer_hook, optimizer=optimizer, scheduler=scheduler))
            
    # return dummies, stepping is done with hooks 
    return LayerWiseDummyOptimizer(), LayerWiseDummyScheduler()

### Setup HF Trainer ###
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, get_constant_schedule
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch, torch.nn as nn, uuid, wandb

lr = 1e-5

# GaLore optimizer hyperparameters
galore_config = dict(
    target_modules_list = ["attn", "mlp"], 
    rank = 1024, 
    update_proj_gap = 200, 
    scale = 2, 
    proj_type="std"
)

modelpath = "meta-llama/Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    torch_dtype=torch.bfloat16,
    attn_implementation = "flash_attention_2",  
    device_map = "auto",
    use_cache = False,
)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False)

# Setup for ChatML
model, tokenizer = setup_chat_format(model, tokenizer)
if tokenizer.pad_token in [None, tokenizer.eos_token]: 
    tokenizer.pad_token = tokenizer.unk_token

# subset of the Open Assistant 2 dataset, 4000 of the top ranking conversations
dataset = load_dataset("g-ronimo/oasst2_top4k_en")

training_arguments = TrainingArguments(
    output_dir = f"out_{run_id}",
    evaluation_strategy = "steps",
    label_names = ["labels"],
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 1,
    save_steps = 250,
    eval_steps = 250,
    logging_steps = 1, 
    learning_rate = lr,
    num_train_epochs = 3,
    lr_scheduler_type = "constant",
    gradient_checkpointing = True,
    group_by_length = False,
)

optimizers = load_galore_optimizer(model, lr, galore_config)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset['test'],
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template = "<|im_start|>user", 
        response_template = "<|im_start|>assistant", 
        tokenizer = tokenizer, 
        mlm = False),
    max_seq_length = 256,
    dataset_kwargs = dict(add_special_tokens = False),
    optimizers = optimizers,
    args = training_arguments,
)

trainer.train()

"""
target_modules_list: Specifies the layers targeted by GaLore
rank: The rank of the projection matrices. Similar to LoRA, the higher the rank the more closely 
      the finetuning will resemble a full parameter finetune. The GaLore authors recomment 1024 for a 7B model.
update_proj_gap: The number of steps after which the projections are updated. 
                 The update is an expensive step and takes around 15 minutes for a 7B model. 
                 Defines the interval for updating projections, with a suggested range between 50 and 1000 steps.
scale: A scale factor akin to LoRA’s alpha, adjusting the update strength.
       After trying a few values I found scale=2 to most closely resemble a classic full-parameter finetune.
"""
