### From https://ertugruldemir.medium.com/reasoning-models-at-home-760310732f6b

import re
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

from nltk.tokenize import word_tokenize
from unsloth import FastLanguageModel, PatchFastRL

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
[Break and reason your answer here]
</reasoning>
<validate>
[Criticize your reason and think about it. If you see somethin start with 'wait']
</validate>
<answer>
[Final integer answer justified by validate]
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_xml_answer(response, tag="answer"):
    import re
    match = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def custom_reward_func(prompts, completions, answer, min_reasoning_length=10, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses_answer = [extract_xml_answer(r, tag="answer") for r in responses]
    extracted_responses_reasoning = [extract_xml_answer(r, tag="reasoning") for r in responses]
    extracted_responses_validate = [extract_xml_answer(r, tag="validate") for r in responses]

    rewards = []
    for original_response, extracted_answer, extracted_reasoning, extracted_validate in zip(
        responses, extracted_responses_answer, extracted_responses_reasoning, extracted_responses_validate
    ):
        is_correct = (extracted_answer == answer[0])
        is_int = extracted_answer.isdigit()
        has_answer_tags = "<answer>" in original_response and "</answer>" in original_response
        has_reasoning_tags = "<reasoning>" in original_response and "</reasoning>" in original_response
        has_validate_tags = "<validate>" in original_response and "</validate>" in original_response
        reasoning_length = len(word_tokenize(extracted_reasoning.lower()))
        validate_length = len(word_tokenize(extracted_validate.lower()))

        reward = 0.0
        reasoning_reward = 0.0

        if is_correct:
            reward += 5.0
        if is_int:
            reward += 0.5

        if has_validate_tags:
            reward *= 1.25
            if validate_length >= 5:
                min_validate_length = 5
                max_validate_length = 256
                max_validate_bonus = 3.0
                if validate_length >= min_validate_length:
                    if validate_length >= max_validate_length:
                        validate_bonus = max_validate_bonus
                    else:
                        validate_bonus = ((validate_length - min_validate_length) / (max_validate_length - min_validate_length)) * max_validate_bonus
                else:
                    validate_bonus = 0.0
            else:
                validate_bonus = 0.0
        else:
            validate_bonus = 0.0

        if has_reasoning_tags:
            reward *= 1.25
            if reasoning_length >= 5:
                min_scaling_length = 5
                max_scaling_length = 1024
                max_scaling_bonus = 10
                if reasoning_length <= min_scaling_length:
                    reasoning_reward = 0.0
                elif reasoning_length >= max_scaling_length:
                    reasoning_reward = 5.0
                else:
                    reasoning_reward = ((reasoning_length - min_scaling_length) / (max_scaling_length - min_scaling_length)) * max_scaling_bonus
            else:
                reasoning_reward = 0.0
        else:
            reasoning_reward = 0.0

        total_reward = reward + reasoning_reward + validate_bonus

        if has_validate_tags:
            validate_lower = extracted_validate.lower()
            if re.search(r"(wait|but|rethink)(?=.{20,})", validate_lower, re.DOTALL):
                total_reward *= 10.0

        rewards.append(total_reward)

    return rewards

dataset = get_gsm8k_questions()
PatchFastRL("GRPO", FastLanguageModel)


max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state = 1907,
)

output_dir="outputs/Qwen-.5B-GRPO"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=3e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_steps=5,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_generations=2,
    max_prompt_length=256,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=1.0,
    report_to="none",
    log_on_each_node=False,
    use_vllm=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[custom_reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
