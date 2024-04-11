# From https://medium.com/towards-artificial-intelligence/few-shots-at-a-math-assistant-with-orca-2-7b-f60a15fe5dfe
# Below code face with error becuase version probelm between transformer and pytorch when bnb_config. Will have to see more

!pip install git+https://github.com/huggingface/transformers
!pip install accelerate -qq
!pip install SentencePiece -qq
!pip install protobuf -qq
!pip install bitsandbytes -qq

import torch
import transformers
from transformers import BitsAndBytesConfig, GenerationConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "microsoft/Orca-2-7b",
    device_map='auto',
    quantization_config=bnb_config)

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/Orca-2-7b",
        use_fast=False,
    )

system_message = """You are Orca, an AI language model created by Microsoft. You are a cautious assistant.

Analyse the maths or logical question given to you and solve it in a step by step manner
"""

user_message = "how many ways can I arrange 10 men in a row?"
prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
inputs = tokenizer(prompt, return_tensors='pt')

from transformers import GenerationConfig

generation_config = GenerationConfig.from_pretrained("microsoft/Orca-2-7b")
generation_config.temperature = 0.1
generation_config.do_sample = True
generation_config.top_p = 0.9

output_ids = model.generate(inputs["input_ids"],generation_config)
answer = tokenizer.batch_decode(output_ids)[0]



