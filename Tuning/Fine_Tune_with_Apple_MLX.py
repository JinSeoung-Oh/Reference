## From https://medium.com/rahasak/fine-tune-llm-for-medical-diagnosis-prediction-with-apple-mlx-1366ca2c5d63

## Setup
# used repository called mlxm
❯❯ git clone https://gitlab.com/rahasak-labs/mlxm.git
❯❯ cd mlxm


# create and activate virtial enviroument 
❯❯ python -m venv .venv
❯❯ source .venv/bin/activate


# install mlx
❯❯ pip install -U mlx-lm


# install other requried pythong pakcages
❯❯ pip install pandas
❯❯ pip install pyarrow

# setup account in hugging-face from here
https://huggingface.co/welcome


# create access token to read/write data from hugging-face through the cli
# this token required when login to huggingface cli
https://huggingface.co/settings/tokens


# setup hugginface-cli
❯❯ pip install huggingface_hub
❯❯ pip install "huggingface_hub[cli]"


# login to huggingface through cli
# it will ask the access to
❯❯ huggingface-cli login

import pandas as pd
import json
import random

# load csv data
file_path = './s2d.csv'
df = pd.read_csv(file_path)

# create text type data
jsonl_data = []
for _, row in df.iterrows():
    diagnosis = row['label']
    symptoms = row['text']
    prompt = f"You are a medical diagnosis expert. You will give patient symptoms: '{symptoms}'. Question: 'What is the diagnosis I have?'. Response: You may be diagnosed with {diagnosis}."
    jsonl_data.append({"text": prompt})

# shuffle the data
random.shuffle(jsonl_data)

# calculate split indices
total_records = len(jsonl_data)
train_split = int(total_records * 2 / 3)
test_split = int(total_records * 1 / 6)

# split the data
train_data = jsonl_data[:train_split]
test_data = jsonl_data[train_split:train_split + test_split]
valid_data = jsonl_data[train_split + test_split:]

# write to JSONL files
with open('train.jsonl', 'w') as train_file:
    for entry in train_data:
        train_file.write(json.dumps(entry) + '\n')

with open('test.jsonl', 'w') as test_file:
    for entry in test_data:
        test_file.write(json.dumps(entry) + '\n')

with open('valid.jsonl', 'w') as valid_file:
    for entry in valid_data:
        valid_file.write(json.dumps(entry) + '\n')

print("data successfully saved to train.jsonl, test.jsonl, and valid.jsonl")


# download llm
❯❯ huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
/Users/lambda.eranga/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/1296dc8fd9b21e6424c9c305c06db9ae60c03ace


# model is downloaded into ~/.cache/huggingface/hub/
❯❯ ls ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2
blobs     refs      snapshots


# list all downloaded models from huggingface
❯❯ huggingface-cli scan-cache

❯❯ python -m mlx_lm.lora \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data data \
  --train \
  --batch-size 4\
  --lora-layers 16\
  --iters 1000


