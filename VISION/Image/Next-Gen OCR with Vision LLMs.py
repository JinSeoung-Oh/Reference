## From https://generativeai.pub/next-gen-ocr-with-vision-llms-a-guide-to-using-phi-3-claude-and-gpt-4o-4c6fbabe92c8

# 1. Using Phi-3 Vision for OCR
"""
conda create -n llm_images python=3.10
conda activate llm_images
pip install torch==2.3.0 torchvision==0.18.0
pip install packaging
pip install pillow==10.3.0 chardet==5.2.0 flash_attn==2.5.8 accelerate==0.30.1 bitsandbytes==0.43.1 requests==2.31.0 transformers==4.40.2 albumentations==1.3.1 opencv-contrib-python==4.10.0.84 matplotlib==3.9.0
pip uninstall jupyter
conda install -c anaconda jupyter
conda update jupyter
pip install --upgrade 'nbconvert>=7' 'mistune>=2'
pip install cchardet
"""

# Import necessary libraries
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
from IPython.display import display
import time

# Define model ID
model_id = "microsoft/Phi-3-vision-128k-instruct"
# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# Define BitsAndBytes configuration for 4-bit quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# Load model with 4-bit quantization and map to CUDA
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    quantization_config=nf4_config,
)

def model_inference(messages, image, max_token):
    start_time = time.time()
    
    # Prepare prompt with image token
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process prompt and image for model input
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    # Generate text response using model
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=max_token,
        do_sample=False,
    )
    # Remove input tokens from generated response
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    # Decode generated IDs to text
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    display(image)
    end_time = time.time()
    print("Inference time: {}".format(end_time - start_time))
    # Print the generated response
    return response

# Define the prompt for extracting information from the front face of the Italian Identity card
prompt = [{"role": "user", "content": "\nOCR the text of the image."}]

# Load image from local path
path_image = "path/to/your/image/image.jpg"  # Replace with your actual image path
image = Image.open(path_image)
# Perform inference
model_inference(prompt, image, 500)
####################################################################################################################

# 2.Using Claude for OCR
"""
pip install anthropic
pip install httpx
"""

import anthropic
import os
import base64
import httpx
# Define the image URL
image_url = "https://path/to/your/image/Haiku.jpg"
image_media_type = "image/jpeg"
# Encode the image in base64
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
# Define the API key (ensure it's set in your environment variables)
api_key = os.getenv("anthropic_key")
# Initialize the client
client = anthropic.Anthropic(api_key=api_key)

def model_inference_claude(prompt_text, image_data, image_media_type):
    # Create the request payload
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ],
        }
    ]
    # Perform inference
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=messages
    )
    # Extract and return the generated response
    return response.content[0].text

# Define the prompt for extracting information
prompt_text = "Describe this image."
# Perform inference
response_text = model_inference_claude(prompt_text, image_data, image_media_type)
# Print the generated response
print(response_text)
####################################################################################################################

# 3.Using GPT-4O for OCR
"""
pip install azure-ai-openai
pip install requests
pip install base64
"""
import base64
from azure.ai.openai import AzureOpenAI
from mimetypes import guess_type
# Define your Azure OpenAI resource endpoint and key
api_base = 'https://YOUR_RESOURCE_NAME.openai.azure.com/'
api_key = 'your_azure_openai_key'
deployment_name = 'your_deployment_name'
api_version = '2024-02-15-preview'  # This might change in the future
# Create a client object
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}openai/deployments/{deployment_name}"
)

# Function to encode a local image into a data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found
# Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"
# Example usage
image_path = 'path/to/your/image.jpg'
data_url = local_image_to_data_url(image_path)

def model_inference_gpt4o(prompt_text, image_data_url):
    # Prepare the request payload
    messages = [
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": [
            { "type": "text", "text": prompt_text },
            { "type": "image_url", "image_url": { "url": image_data_url } }
        ] }
    ]
# Perform inference
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=2000
    )
    # Extract and return the generated response
    return response.choices[0].message['content']
# Example usage
prompt_text = "Describe this picture:"
response_text = model_inference_gpt4o(prompt_text, data_url)
# Print the generated response
print(response_text)

############################################################################
# Import necessary libraries
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
from IPython.display import display
import time
import anthropic
import os
import base64
import httpx
from azure.ai.openai import AzureOpenAI
from mimetypes import guess_type

# Configuration for Phi-3 Vision
phi3_model_id = "microsoft/Phi-3-vision-128k-instruct"
phi3_processor = AutoProcessor.from_pretrained(phi3_model_id, trust_remote_code=True)
phi3_nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
phi3_model = AutoModelForCausalLM.from_pretrained(
    phi3_model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    quantization_config=phi3_nf4_config,
)

# Configuration for Claude
claude_api_key = os.getenv("anthropic_key")
claude_client = anthropic.Anthropic(api_key=claude_api_key)

# Configuration for GPT-4O
api_base = 'https://YOUR_RESOURCE_NAME.openai.azure.com/'
api_key = 'your_azure_openai_key'
deployment_name = 'your_deployment_name'
api_version = '2024-02-15-preview'
gpt4o_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}openai/deployments/{deployment_name}"
)

# Function for Phi-3 Vision inference
def model_inference_phi3(messages, image):
    start_time = time.time()
    prompt = phi3_processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = phi3_processor(prompt, [image], return_tensors="pt").to("cuda:0")
    generate_ids = phi3_model.generate(
        **inputs,
        eos_token_id=phi3_processor.tokenizer.eos_token_id,
        max_new_tokens=500,
        do_sample=False,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = phi3_processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    display(image)
    end_time = time.time()
    print("Inference time: {}".format(end_time - start_time))
    print(response)

# Function for Claude inference
def model_inference_claude(prompt_text, image_data, image_media_type):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ],
        }
    ]
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=messages
    )
    return response.content[0].text

# Function for GPT-4O inference
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def model_inference_gpt4o(prompt_text, image_data_url):
    messages = [
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": [
            { "type": "text", "text": prompt_text },
            { "type": "image_url", "image_url": { "url": image_data_url } }
        ] }
    ]
    response = gpt4o_client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=2000
    )
    return response.choices[0].message['content']

# Example usage
# Phi-3 Vision
phi3_prompt = [{"role": "user", "content": "\nOCR the text of the image. Extract the text of the following fields and put it in a JSON format: 'Comune Di/ Municipality', 'COGNOME /Surname', 'NOME/NAME', 'LUOGO E DATA DI NASCITA/PLACE AND DATE OF BIRTH', 'SESSO/SEX', 'STATURA/HEIGHT', 'CITADINANZA/NATIONALITY', 'EMISSIONE/ ISSUING', 'SCADENZA /EXPIRY'. Read the code at the top right and put it in the JSON field 'CODE'"}]
phi3_image_path = "path/to/your/image/cie_fronte.jpg"
phi3_image = Image.open(phi3_image_path)
model_inference_phi3(phi3_prompt, phi3_image)

# Claude
claude_prompt = "Describe this image."
claude_image_url = "https://upload.wikimedia.org/wikipedia/commons/1/12/Haiku_de_L._M._Panero.jpg"
claude_image_media_type = "image/jpeg"
claude_image_data = base64.b64encode(httpx.get(claude_image_url).content).decode("utf-8")
claude_response = model_inference_claude(claude_prompt, claude_image_data, claude_image_media_type)
print(claude_response)

# GPT-4O
gpt4o_prompt = "Describe this picture:"
gpt4o_image_path = "path/to/your/image.jpg"
gpt4o_data_url = local_image_to_data_url(gpt4o_image_path)
gpt4o_response = model_inference_gpt4o(gpt4o_prompt, gpt4o_data_url)
print(gpt4o_response)







