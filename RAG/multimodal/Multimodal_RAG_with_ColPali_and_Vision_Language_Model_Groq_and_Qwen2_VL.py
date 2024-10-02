## From https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd

!pip install -qU byaldi
!pip install -qU accelerate
!pip install -qU flash_attn
!pip install -qU qwen_vl_utils
!pip install -qU pdf2image
!pip install -qU groq
!python -m pip install git+https://github.com/huggingface/transformers

!sudo apt-get update
!apt-get install poppler-utils

!mkdir Data
!wget https://arxiv.org/pdf/2409.06697 -O Data/input.pdf

from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from IPython.display import Image,display
import torch
from pdf2image import convert_from_path
import groq
from google.colab import userdata
import os
from groq import Groq
import base64
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", 
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation="flash_attention_2",
                                                        device_map="cuda")
RAG.index(input_path="Data/input.pdf",
          index_name="multimodal_rag",
          store_collection_with_index=False,
          overwrite=True,)
text_query = "What is the type of star hosting thge kepler-51 planetary system?"
results = RAG.search(text_query,k=3)

--------------------------------------------------------------------------------------------------------
### Convert to actual Image Data
images = convert_from_path("Data/input.pdf")
image_index = results[0]["page_num"] -1

### display the Chosen Document Image
from IPython.display import Image,display
display(images[image_index])
--------------------------------------------------------------------------------------------------------
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/content/image1.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llava-v1.5-7b-4096-preview",
)

--------------------------------------------------------------------------------------------------------
## Using QWEN2-VL-7B-INSTRUCT Vision Language model for response synthesis
messages = [
    {"role":"user",
     "content":[{"type":"image",
                 "image":images[image_index]
                 },
                {"type":"text","text":text_query}
              ]
    }
            ]

#
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#
image_inputs,video_inputs = process_vision_info(messages)
#
inputs = processor(text=[text],
                   images=image_inputs,
                   videos=video_inputs,
                   padding=True,
                   return_tensors="pt")
inputs = inputs.to("cuda")
#
generate_ids = model.generate(**inputs, 
                              max_new_tokens=256)
#
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
#
output_text = processor.batch_decode(generated_ids_trimmed, 
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
#
print(output_text[0])

--------------------------------------------------------------------------------------------------------
## ask another query
text_query  = "What is the age of the star hosting the kepler-51 planetary system?"
#
messages = [
    {"role":"user",
     "content":[{"type":"image",
                 "image":images[image_index]
                 },
                {"type":"text","text":text_query}
              ]
    }
            ]

#
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#
image_inputs,video_inputs = process_vision_info(messages)
#
inputs = processor(text=[text],
                   images=image_inputs,
                   videos=video_inputs,
                   padding=True,
                   return_tensors="pt")
inputs = inputs.to("cuda")
#
generate_ids = model.generate(**inputs, 
                              max_new_tokens=256)
#
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
#
output_text = processor.batch_decode(generated_ids_trimmed, 
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
#
print(output_text[0])








