### From https://ai.plainenglish.io/fine-tuning-googles-gemma-3-12b-for-reasoning-how-grpo-turned-a-good-model-into-a-brilliant-db8c272c67ea

from huggingface_hub import notebook_login
notebook_login()

!pip install -qqq git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 \
                  git+https://github.com/huggingface/trl.git@main \
                  bitsandbytes


import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-3-4b-it", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
)
lora_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(model, lora_config)
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
tokenizer = processor.tokenizer

SYSTEM_PROMPT = "Respond in structured reasoning format (XML)."
def get_gsm8k_questions(split="train"):
    data = load_dataset('openai/gsm8k', 'main')[split]
    return data.map(lambda x: {
        'prompt': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': x['question']}],
        'answer': x['answer']
    })
train_data = get_gsm8k_questions()

def correctness_reward_func(prompts, completions, answer):
    responses = [extract_xml_answer(c[0]['content']) for c in completions]
    return [2.0 if r == a else 0.0 for r, a in zip(responses, answer)]

merged_model = model.merge_and_unload()
merged_model.push_to_hub("your-username/gemma-reasoning-genius")
