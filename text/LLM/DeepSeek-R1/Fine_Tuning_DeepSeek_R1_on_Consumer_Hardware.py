### From https://medium.com/@pankaj_pandey/fine-tuning-deepseek-r1-on-c

"""
Common Challenges in Fine-Tuning and How to Overcome Them
1. Computational Limitations 💾⚙️🚀
   Challenge: Fine-tuning LLMs requires high-end GPUs with significant VRAM and memory resources.
   Solution: Use LoRA and 4-bit quantization to reduce computational load. Offloading certain processes to CPU or cloud-based services like Google Colab or AWS can also help.
2. Overfitting on Small Datasets 🎯📊📉
   Challenge: Training on a small dataset may cause the model to memorize responses instead of generalizing well.
   Solution: Use data augmentation techniques and regularization methods like dropout or early stopping to prevent overfitting.
3. Long Training Times ⏳⚡🔧
   Challenge: Fine-tuning can take days or weeks depending on hardware and dataset size.
   Solution: Utilize gradient checkpointing and low-rank adaptation (LoRA) to speed up training while maintaining efficiency.
4. Catastrophic Forgetting ⚠️📉🚀
   Challenge: The fine-tuned model may forget general knowledge from its pretraining phase.
   Solution: Use a mixed dataset containing both domain-specific data and general knowledge data to maintain overall model accuracy.
5. Bias in Fine-Tuned Models ⚖️🧐🤖
   Challenge: Fine-tuned models can inherit biases present in the dataset.
   Solution: Curate diverse and unbiased datasets, apply debiasing techniques and evaluate the model using fairness metrics.
"""
pip install unsloth torch transformers datasets accelerate bitsandbytes

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import Trainer

model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

dataset = load_dataset("json", data_files={"train": "train_data.jsonl", "test": "test_data.jsonl"})

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}
### Response:
"""
def preprocess_function(examples):
    inputs = [prompt_template.format(instruction=inst) for inst in examples["instruction"]]
    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True)
    return model_inputs
tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "v_proj"],  # Fine-tune key attention layers
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {eval_results['perplexity']}")

# Save the model and tokenizer
model.save_pretrained("./finetuned_deepseek_r1")
tokenizer.save_pretrained("./finetuned_deepseek_r1")

### Deploying the Model for Inference
./llama.cpp/llama-cli \
   --model unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
   --cache-type-k q8_0 \
   --threads 16 \
   --prompt '<|User|>What is 1+1?<|Assistant|>' \
   --n-gpu-layers 20 \
   -no-cnv



