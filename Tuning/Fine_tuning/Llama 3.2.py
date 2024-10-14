### From https://towardsdatascience.com/fine-tune-llama-3-2-for-powerful-performance-in-targeted-domains-8c4fccef93dd

### Prepare dataset
import pandas as pd

def create_random_math_question():
 # select 2 random numbers
 import random
 num1 = random.randint(1, 1000)
 num2 = random.randint(1, 1000)
 res = num1 + num2
 return num1, num2, res

dataset = []
for _ in range(10000):
 num1, num2, res = create_random_math_question()
 prompt = f"What is the answer to the following math question: {num1} + {num2}?"
 dataset.append((prompt, str(res)))

df = pd.DataFrame(dataset, columns=["prompt", "target"])

new_dataset = []
for prompt, response in dataset:
 new_row = [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}]
 new_dataset.append(new_row)

df = pd.DataFrame({'conversations': new_dataset})
df.to_pickle("math_dataset.pkl")

from datasets import Dataset
dataset = Dataset.from_pandas(df)

### Setting up the training script to fine-tune Llama 3.2
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

df = pd.read_pickle("math_dataset.pkl")
dataset = Dataset.from_pandas(df)
dataset = dataset.map(apply_template, batched=True)

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

print("Training")
trainer.train()

now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save_pretrained_merged(f"model_{now}", tokenizer, save_method="merged_16bit")

#### Testing
import pandas as pd

def create_random_math_question():
 # select 2 random numbers
 import random
 num1 = random.randint(1, 1000)
 num2 = random.randint(1, 1000)
 res = num1 + num2
 return num1, num2, res

dataset = []
for _ in range(1000):
 num1, num2, res = create_random_math_question()
 prompt = f"What is the answer to the following math question: {num1} + {num2}?"
 dataset.append((prompt, str(res)))

new_dataset = []
for prompt, response in dataset:
 new_row = [{"from": "human", "value": prompt}, {"gt": response}]
 new_dataset.append(new_row)

df = pd.DataFrame(new_dataset, columns=["prompt", "gt"])
df.to_pickle("math_dataset_test.pkl")
"""
File to run inference on a test sample of documents for new fine-tuned model
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, TextStreamer
from peft import PeftModel
import pandas as pd


SAVED_MODEL_FOLDER = "model" # TODO update this to the folder your main model is saved to
SAVED_ADAPTER_FOLDER = "output/checkpoint-24-own-dataset" # TODO update this to the folder your adapter model is saved to

max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
 model_name=SAVED_MODEL_FOLDER,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.for_inference(model)

model = PeftModel.from_pretrained(model, SAVED_ADAPTER_FOLDER)

df_test = pd.read_pickle("math_dataset_test.pkl")
messages = df_test["prompt"].tolist()

responses = []
for message in messages:
    message = [message] # must wrap in a list
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    response = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=64, use_cache=True)
    
    # the response is a list of tokens, decode those into text
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    responses.append(response_text)

# save responses to pickle
df_test["response_finetuned_model"] = responses

now = pd.Timestamp.now()
df_test.to_pickle(f"math_dataset_test_finetuned_{now}.pkl")

model = PeftModel.from_pretrained(model, SAVED_ADAPTER_FOLDER)


