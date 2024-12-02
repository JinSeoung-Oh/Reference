### From https://generativeai.pub/llm2clip-microsofts-powerful-language-model-unlocking-richer-visual-representations-and-4ef722a915d8

"""
The intersection of multimodal AI systems, exemplified by models like CLIP (Contrastive Language–Image Pretraining), 
and Large Language Models (LLMs) such as GPT-4 and LLaMA, has opened up new frontiers in artificial intelligence.
While CLIP excels at aligning text and image representations for tasks like retrieval and classification,
its textual understanding has limitations, particularly with complex or long captions. 
LLMs, with their extensive reasoning and linguistic comprehension, present a solution. 
LLM2CLIP, a transformative AI framework, leverages the strengths of LLMs to extend CLIP’s capabilities, 
offering unprecedented advancements in multimodal AI tasks.

-1. What is LLM2CLIP?
    LLM2CLIP enhances the traditional CLIP model by integrating fine-tuned LLMs into its architecture, 
    either replacing or complementing its text encoder. This results in richer textual understanding and improved cross-modal alignment.

    -a. Key Features
        Caption Contrastive Fine-Tuning (CC Fine-Tuning):

        -1) Aligns captions of the same image in feature space while pushing unrelated captions apart.
        -2) Enables LLMs to serve as high-quality text encoders, addressing CLIP’s limitations in textual discrimination.

    -b. Rich Textual Understanding:
        -1) Leverages LLMs’ open-world knowledge to process dense, complex, and multilingual captions beyond CLIP’s inherent capabilities.
          
    -c. Efficient Training:
        -1) Freezes LLM gradients to retain knowledge while reducing computational overhead.
        -2) Pre-extracts textual features, optimizing computational efficiency.

    -d. Cross-Lingual Capability:
        -1) Empowers the model to perform tasks in multiple languages, transferring knowledge effectively even when trained solely on English datasets.

-2. How Does LLM2CLIP Work?
    -a. Challenges in Integrating LLMs with CLIP
        -1) LLM Embedding Limitations:
            - LLMs are designed for auto-regressive tasks, making their embeddings unsuitable for CLIP's discriminative tasks.
            - Textual embeddings from LLMs lack the separability needed for effective cross-modal tasks.
        -2) Alignment with CLIP’s Visual Encoder:
            - Mismatches between LLM and CLIP representations hinder direct integration.

    -b. Innovative Solutions
        -1) Caption Contrastive Fine-Tuning:
            - Treats captions of the same image as positive pairs, while unrelated captions are treated as negatives.
            - Uses a supervised contrastive loss to enhance the LLM's ability to differentiate between captions.
        -2) Freezing LLM Gradients:
            - Preserves LLMs' pre-trained knowledge by freezing their gradients.
            - Lightweight adapters align LLM embeddings with CLIP’s visual encoder, maintaining seamless integration.
        -3) Leveraging Open-World Knowledge:
            - Incorporates the LLM’s vast pre-trained knowledge, enabling better contextual understanding and multilingual support.

-3. Applications of LLM2CLIP
    -a. Advanced Image-Text Retrieval
        - Achieves state-of-the-art (SOTA) performance in short and long-text retrieval tasks across datasets like COCO, Flickr30k, and Urban1k.
        - Handles both simple and complex queries effectively, offering broader utility in real-world applications.
    -b. Cross-Lingual Applications
        - Allows English-trained models to excel in tasks involving other languages, such as Chinese, without explicit multilingual training.
        - Surpasses models trained on native datasets, demonstrating superior knowledge transfer.
    -c. Long-Text Understanding
        - Processes captions exceeding CLIP’s 77-token limit.
        - Excels in tasks requiring detailed image descriptions and context-rich queries.
    -d. Multimodal Generalization
        - Lays the foundation for incorporating modalities beyond text and images, including video and audio.
        - Enables comprehensive data integration for future AI systems.
    -e. Enhanced Vision-Language Models
        - Boosts performance in Vision-Language Large Models (VLLMs) like Llava 1.5.
        - Improves tasks such as Visual Question Answering (VQA), benefiting education, accessibility, and more.

-4. Why LLM2CLIP Matters
    -a. Broadens Multimodal AI Horizons
        - Unlocks richer and more nuanced representations, essential for applications in healthcare, autonomous vehicles, 
          and interactive AI systems.
    -b. Cost-Effective Scaling
        - Employs efficient training strategies, making advanced multimodal AI accessible to researchers and organizations 
          with limited computational resources.
    -c. Breaks Language Barriers
        - Revolutionizes cross-lingual learning, enabling AI systems to work effectively across diverse linguistic contexts.
    -d. Future-Proofs AI
        Establishes a scalable framework for integrating LLMs into multimodal systems, paving the way for incorporating diverse 
        modalities like video and 3D data.
"""
pip install llm2vec
git clone https://github.com/microsoft/LLM2CLIP.git && cd LLM2CLIP
cd llm2clip/

pip install -r requirements.txt

conda install -c conda-forge --override-channels notebook
conda install -c conda-forge --override-channels ipywidgets -y jupyter notebook

from PIL import Image
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import CLIPImageProcessor
import torch
from llm2vec import LLM2Vec

# Image processor
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# CLIP model
model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336"
model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True
).to('cuda').eval()

# LLM for captions
llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
llm_model = AutoModel.from_pretrained(llm_model_name, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Workaround for LLM2Vec
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

# Initialize LLM2Vec
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

captions = ["a diagram", "a dog", "horses"]
image_path = "home/user/Download/horses.png"

image = Image.open(image_path)
input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.get_image_features(input_pixels)
    text_features = l2v.encode(captions, convert_to_tensor=True).to('cuda')
    text_features = model.get_text_features(text_features)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute probabilities
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)



