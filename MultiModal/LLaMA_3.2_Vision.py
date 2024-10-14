### From https://pub.towardsai.net/llama-3-2-vision-revolutionizing-multimodal-ai-with-advanced-visual-reasoning-now-llama-can-see-d8a32d8e4b86

!pip install git+https://github.com/huggingface/transformers accelerate bitsandbytes huggingface_hub
!pip install -U "huggingface_hub[cli]"

from huggingface_hub import notebook_login
notebook_login()

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)

# Load an image from a URL
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/rabbit.jpg"

image = Image.open(requests.get(url, stream=True).raw)

# Define the conversation prompt
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Can you please describe this image in just one sentence?"}
    ]},
    {"role": "assistant", "content": "The image depicts a rabbit dressed in a blue coat and brown vest, standing on a dirt road in front of a stone house."},
    {"role": "user", "content": "What is in the background?"}
]

input_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=70)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))
