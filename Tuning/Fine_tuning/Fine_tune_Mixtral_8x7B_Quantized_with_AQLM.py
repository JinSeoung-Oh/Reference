"""
From https://pub.towardsai.net/fine-tune-mixtral-8x7b-quantized-with-aqlm-2-bit-on-your-gpu-4f8fac86e523
https://huggingface.co/collections/ISTA-DASLab/aqlm-65e8dc75b908c7d73ec35598

Not AQLM but it is useful : https://github.com/Cornell-RelaxML/quip-sharp
"""

pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/peft
pip install git+https://github.com/huggingface/trl.git
pip install git+https://github.com/huggingface/accelerate.git
pip install aqlm[gpu,cpu]

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf",
    trust_remote_code=True, torch_dtype="auto", device_map="cuda", low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained("ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf")
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
dataset = load_dataset("timdettmers/openassistant-guanaco")

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate", "w1", "w2", "w3"]
)

training_arguments = TrainingArguments(
        output_dir="./fine-tuned_MixtralAQLM_2bit/",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        log_level="debug",
        logging_steps=25,
        learning_rate=1e-4,
        eval_steps=25,
        save_strategy='steps',
        max_steps=100,
        warmup_steps=25,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        args=training_arguments,
)
trainer.train()



