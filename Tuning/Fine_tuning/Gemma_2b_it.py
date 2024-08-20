## From https://aashi-dutt3.medium.com/part-2-fine-tune-gemma-2b-it-model-a26246c530e7

import os
import transformers
import torch
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer

# Download Gemma 2b-it base model
model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config = bnb_config,
                                             device_map={"":0})

# Configure LoRA
lora_config = LoraConfig(
    r = 8,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM"
)

# Get the data
data = load_dataset("Aashi/Science_Q_and_A_dataset")
data = data.map(lambda samples: tokenizer(samples["Question"], samples["Context"]), batched=True)

def formatting_func(example):
  text = f"Answer: {example['Answer'][0]}"
  return [text]

trainer = SFTTrainer(
    model = model,
    train_dataset = data["train"],
    args = transformers.TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 2,
        max_steps = 75,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "paged_adamw_8bit"
    ),
    peft_config = lora_config,
    formatting_func = formatting_func

)

trainer.train()

#### Save model
fine_tuned_model = "fine_tuned_science_gemma2b-it_unmerged"
trainer.model.save_pretrained(fine_tuned_model)

# Push the model on Hugging Face.
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype = torch.float16,
    device_map = {"": 0}
)

# Merge the fine-tuned model with LoRA adaption along with the base Gemma 2b-it model.
fine_tuned_merged_model = PeftModel.from_pretrained(base_model, fine_tuned_model)
fine_tuned_merged_model = fine_tuned_merged_model.merge_and_unload()

# Save the fine-tuned merged model.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True)
fine_tuned_merged_model.save_pretrained("fine_tuned_science_gemma2b-it", safe_serialization = True)
tokenizer.save_pretrained("fine_tuned_science_gemma2b-it")
tokenizer.padding_side = "right"

#### Inference
text = "What is Hemoglobin?"

device = "cuda:0"

prompt = text + "\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)




