### from https://blog.gopenai.com/fine-tuning-llm-model-meta-llama-3-1-8b-for-text-to-sql-data-ea5a07620dd3

from unsloth import FastLanguageModel
import torch

Configuration = """
max_seq_length = 2048 # we support rope scaling internally
dtype = None
load_in_4bit = True # 4 bit quantization to reduce memory
"""

# Loading the pre-trained model and tokenizer with the specified configuration.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Applies Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA (Low-Rank Adaptation) to the model.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [...],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Template for generating SQL responses based on input questions and context.
sql_prompt = """Below is input question that user ask, context is given to help user's question, generate SQL response for user's question.

### Input:
{}

### Context:
{}

### SQL Response:
{}"""


def formatting_prompts_func(examples):
    instructions = examples["context"] # note : these are my columns 
    inputs = examples["question"] # this one
    outputs = examples["sql_query"] # and this one 
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts,}

from datasets import load_dataset
import pandas as pd

df = pd.read_excel('fine-tuning-dataset_latest.xlsx')
from datasets import Dataset
dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']
dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(  # here we defined the TrainingArguments
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer = trainer.train()
trainer.model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
print("done")

