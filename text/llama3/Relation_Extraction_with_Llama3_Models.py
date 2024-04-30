# https://towardsdatascience.com/relation-extraction-with-llama3-models-f8bc41858b9e

!pip install -q groq
!pip install -U accelerate bitsandbytes datasets evaluate 
!pip install -U peft transformers trl 

# For Google Colab settings
from google.colab import userdata, drive

# This will prompt for authorization
drive.mount('/content/drive')

# Set the working directory
%cd '/content/drive/MyDrive/postedBlogs/llama3RE'

# For Hugging Face Hub setting
from huggingface_hub import login

# Upload the HuggingFace token (should have WRITE access) from Colab secrets
HF = userdata.get('HF')

# This is needed to upload the model to HuggingFace
login(token=HF,add_to_git_credential=True)

# Create a path variable for the data folder
data_path = '/content/drive/MyDrive/postedBlogs/llama3RE/datas/'

# Full fine-tuning dataset
sft_dataset_file = f'{data_path}sft_train_data.json'

# Data collected from the the mini-test
mini_data_path = f'{data_path}mini_data.json'

# Test data containing all three outputs
all_tests_data = f'{data_path}all_tests.json'

# The adjusted training dataset
train_data_path = f'{data_path}sft_train_data.json'

# Create a path variable for the SFT model to be saved locally
sft_model_path = '/content/drive/MyDrive/llama3RE/Llama3_RE/'

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Choose the desired category from the dataset
ie_category = [e for e in dataset["train"] if e["category"]=="information_extraction"]

# Retain only the context from each instance
ie_context = [e["context"] for e in ie_category]

# Split the text into sentences (at the period) and keep the first sentence
reduced_context = [text.split('.')[0] + '.' for text in ie_context]

# Retain sequences of specified lengths only (use character length)
sampler = [e for e in reduced_context if 30 < len(e) < 170]

system_message = """You are an experienced annontator. 
Extract all entities and the relations between them from the following text. 
Write the answer as a triple entity1|relationship|entitity2. 
Do not add anything else.
Example Text: Alice is from France.
Answer: Alice|is from|France.
"""

messages = [[
    {"role": "system","content": f"{system_message}"},
    {"role": "user", "content": e}] for e in sampler]

###################### The Groq Client and API #################################
import os
from groq import Groq

gclient = Groq(
    api_key=userdata.get("GROQ"),
)

import time
from tqdm import tqdm

def process_data(prompt):

    """Send one request and retrieve model's generation."""

    chat_completion = gclient.chat.completions.create(
        messages=prompt, # input prompt to send to the model
        model="llama3-70b-8192", # according to GroqCloud labeling
        temperature=0.5, # controls diversity
        max_tokens=128, # max number tokens to generate
        top_p=1, # proportion of likelihood weighted options to consider
        stop=None, # string that signals to stop generating
        stream=False, # if set partial messages are sent
    )
    return chat_completion.choices[0].message.content


def send_messages(messages):

    """Process messages in batches with a pause between batches."""
   
   batch_size = 10
    answers = []

    for i in tqdm(range(0, len(messages), batch_size)): # batches of size 10

        batch = messages[i:i+10]  # get the next batch of messages

        for message in batch:
            output = process_data(message)
            answers.append(output)

        if i + 10 < len(messages):  # check if there are batches left
            time.sleep(10)  # wait for 10 seconds

    return answers

# Data generation with Llama3-70B
answers = send_messages(messages)

# Combine input data with the generated dataset
combined_dataset = [{'text': user, 'gold_re': output} for user, output in zip(sampler, answers)]

################### Supervised Fine Tuning of Llama3–8B #################################
def create_conversation(sample):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": sample["text"]},
            {"role": "assistant", "content": sample["gold_re"]}
        ]
    }

from datasets import load_dataset, Dataset

train_dataset = Dataset.from_list(train_data)

# Transform to conversational format
train_dataset = train_dataset.map(create_conversation,
                      remove_columns=train_dataset.features,
                      batched=False)

model_id  =  "meta-llama/Meta-Llama-3-8B"

from transformers import AutoTokenizer

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          use_fast=True,
                                          trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
tokenizer.padding_side = 'left'

# Set a maximum length
tokenizer.model_max_length = 512

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from trl import setup_chat_format

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config
)

model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig

# According to Sebastian Raschka findings
peft_config = LoraConfig(
        lora_alpha=128, #32
        lora_dropout=0.05,
        r=256,  #16
        bias="none",
        target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", 
          "down_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
)

######################## Training Arguments #############################
from transformers import TrainingArguments

# Adapted from  Phil Schmid blogpost
args = TrainingArguments(
    output_dir=sft_model_path,              # directory to save the model and repository id
    num_train_epochs=2,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory, use in distributed training
    optim="adamw_8bit",                     # choose paged_adamw_8bit if not enough memory
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to Hugging Face hub
    hub_model_id="llama3-8b-sft-qlora-re",
    report_to="tensorboard",               # report metrics to tensorboard
    )

####################### Initialize the Trainer and Train the Model ###################################
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=sft_dataset,
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=False, # True if the dataset is large
    dataset_kwargs={
        "add_special_tokens": False,  # the template adds the special tokens
        "append_concat_token": False, # no need to add additional separator token
    }
)

trainer.train()
trainer.save_model()

import torch
import gc
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

############################# Inference with SFT Model ###############################
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
import torch

# HF model
peft_model_id = "solanaO/llama3-8b-sft-qlora-re"

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16,
  offload_buffers=True
)

okenizer = AutoTokenizer.from_pretrained(peft_model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def create_input_prompt(sample):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": sample["text"]},
        ]
    }
    
from datasets import Dataset

test_dataset = Dataset.from_list(mini_data)

# Transform to conversational format
test_dataset = test_dataset.map(create_input_prompt,
                      remove_columns=test_dataset.features,
                      batched=False)

############################ One Sample Test ##################################
# Generate the input prompt
prompt = pipe.tokenizer.apply_chat_template(test_dataset[2]["messages"][:2],
                                            tokenize=False,
                                            add_generation_prompt=True)
# Generate the output
outputs = pipe(prompt,
              max_new_tokens=128,
              do_sample=False,
              temperature=0.1,
              top_k=50,
              top_p=0.1,
              )
# Display the results
print(f"Question: {test_dataset[2]['messages'][1]['content']}\n")
print(f"Gold-RE: {test_sampler[2]['gold_re']}\n")
print(f"LLama3-8B-RE: {test_sampler[2]['test_re']}\n")
print(f"SFT-Llama3-8B-RE: {outputs[0]['generated_text'][len(prompt):].strip()}")


