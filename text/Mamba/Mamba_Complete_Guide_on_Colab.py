## From https://medium.com/@scholarly360/mamba-complete-guide-on-colab-cd73811e8f47
## https://colab.research.google.com/drive/1jDlFa8RQ_ByzwN-3DkSU4umJGSgRu4Nu?usp=drive_link&source=post_page-----cd73811e8f47--------------------------------

!pip install causal-conv1d==1.0.0
!pip install mamba-ssm==1.0.1

## state-spaces/mamba-2.8b

import torch
import os
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(os.path.expanduser("state-spaces/mamba-2.8b"), device="cuda", dtype=torch.bfloat16)

tokens = tokenizer("What is the meaning of life", return_tensors="pt")
input_ids = tokens.input_ids.to(device="cuda")
max_length = input_ids.shape[1] + 80
fn = lambda: model.generate(
        input_ids=input_ids, max_length=max_length, cg=True,
        return_dict_in_generate=True, output_scores=True,
        enable_timing=False, temperature=0.1, top_k=10, top_p=0.1,)
out = fn()
print(tokenizer.decode(out[0][0]))


## havenhq/mamba-chat

import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("havenhq/mamba-chat")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

model = MambaLMHeadModel.from_pretrained("havenhq/mamba-chat", device="cuda", dtype=torch.float16)


messages = []
user_message = """
What is the date for announcement
On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore.
 """

messages.append(dict(role="user",content=user_message))
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
out = model.generate(input_ids=input_ids, max_length=2000, temperature=0.9, top_p=0.7, eos_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(out)
messages.append(dict(role="assistant",content=decoded[0].split("<|assistant|>\n")[-1]))
print("Model:", decoded[0].split("<|assistant|>\n")[-1])


