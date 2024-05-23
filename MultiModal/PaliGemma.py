"""
From https://blog.roboflow.com/paligemma-multimodal-vision/

# What is PaliGemma?
PaliGemma, released alongside other products at the 2024 Google I/O event, is a combined multimodal model based on two other models 
from Google research: SigLIP, a vision model, and Gemma, a large language model, 
which means the model is a composition of a Transformer decoder and a Vision Transformer image encoder.
It takes both image and text as input and generates text as output, supporting multiple languages.

# Important aspects of PaliGemma:
- Relatively small 3 billion combined parameter model
- Permissible commercial use terms
- Ability to fine-tune for image and short video caption, visual question answering, text reading, object detection, and object segmentation
While PaliGemma is useful without fine-tuning, Google says it is “not designed to be used directly, but to be transferred (by fine-tuning) 
to specific tasks using a similar prompt structure” which means whatever baseline we can observe 
with the model weights is only the tip of the iceberg for how useful the model may be in a given context. 
PaliGemma is pre-trained on WebLI, CC3M-35L, VQ²A-CC3M-35L/VQG-CC3M-35L, OpenImages, and WIT.

# Links to PaliGemma Resources
Google supplied ample resources to start prototyping with PaliGemma and we’ve curated the highest quality information for those of you who want 
to jump into using PaliGemma immediately. 

- PaliGemma Github README
- PaliGemma documentation
- PaliGemma fine-tuning documentation
- Fine-tune PaliGemma in Google Colab
 - Access PaliGemma in Google Vertex
In this post we will explore what PaliGemma can do, compare PaliGemma benchmarks to other LMMs, understand PaliGemma’s limitations, 
and see how it performs in real world use cases. We’ve put together learnings that can save you time while testing PaliGemma.

Let’s get started!

What can PaliGemma do?
PaliGemma is a single-turn vision language model and it works best when fine-tuning to a specific use case. 
This means you can input an image and text string, such as a prompt to caption the image, or a question and PaliGemma will output text in response to the input,
such as a caption of the image, an answer to a question, or a list of object bounding box coordinates.

Tasks PaliGemma is suited to perform relate to the benchmarking results Google released across the following tasks:

- Fine-tuning on single tasks
- Image question answering and captioning
- Video question answering and captioning
- Segmentation

How to Fine-tune PaliGemma
One of the exciting aspects of PaliGemma is its ability to finetune on custom use-case data.
A notebook published by Google’s PaliGemma team showcases how to fine-tune on a small dataset.
"""

# How to Deploy and Use PaliGemma
!git clone https://github.com/roboflow/inference.git
%cd inference
!pip install -e .
!pip install git+https://github.com/huggingface/transformers.git accelerate -q

import inference
from inference.models.paligemma.paligemma import PaliGemma

pg = PaliGemma(api_key="YOUR ROBOFLOW API KEY")
from PIL import Image

image = Image.open("/content/dog.webp") # Change to your image
prompt = "How many dogs are in this image?"

result = pg.predict(image,prompt)
