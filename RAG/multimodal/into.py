"""
## From https://towardsdatascience.com/multimodal-rag-intuitively-and-exhaustively-explained-5713d8069eb0

1. Multimodal RAG Overview
   -1. Concept
       Multimodal RAG extends traditional RAG systems by incorporating multiple forms of information, such as text, images, and videos, into a multimodal model.
       Instead of retrieving only text based on a user's prompt, a multimodal RAG system can retrieve and utilize data from various modalities.
   -2. Purpose
       The goal is to enhance the capabilities of language models by allowing them to process and integrate diverse data types, 
       leading to richer and more informative outputs.

2. Approaches to Multimodal RAG
  -1. Shared Vector Space
      - Description
        This approach uses an embedding that works across multiple modalities, similar to the CLIP model. 
        Data is processed through encoders that are designed to integrate seamlessly, 
        allowing retrieval of the most similar data across all modalities in response to a user's query.
      - Implementation
        Systems like Google's Vertex AI provide multimodal embedding solutions, placing data from different modalities into a common embedding space.

  -2. Single Grounded Modality
      - Description
        In this approach, all data modalities are converted into a single modality, typically text, before being processed by a single encoder.
      - Advantages
        This method simplifies processing and reduces complexity while maintaining high-quality results in many applications.
      - Challenges
        There is a theoretical risk of losing subtle information during the conversion process, but practical implementations often mitigate this risk effectively.

   -3. Separate Retrieval
       - Description
         This approach involves using a collection of models, each designed to work with specific modalities. Retrieval is performed separately for each modality, 
         and the results are combined.
       - Implementation
         The combined retrievals can be simply aggregated and fed into a multimodal model, or more sophisticated methods
         like re-ranking can be used to organize data based on its relevance to the query.
       - Benefits
         This approach is flexible, allowing optimization for different modalities and accommodating modalities not supported by existing models.
"""

!pip install pydub

import os
import google.generativeai as genai
from google.colab import userdata
import requests
from PIL import Image
from IPython.display import display
from pydub import AudioSegment
import numpy as np
import io
import matplotlib.pyplot as plt
import wave

os.environ["GOOGLE_API_KEY"] = userdata.get('GeminiAPIKey')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Downloading image file
url = 'https://github.com/DanielWarfield1/MLWritingAndResearch/blob/main/Assets/Multimodal/MMRAG/Lorenz_Ro28-200px.png?raw=true'
response = requests.get(url, stream=True)
image = Image.open(response.raw).convert('RGB')

# Downloading audio file
url = "https://github.com/DanielWarfield1/MLWritingAndResearch/blob/main/Assets/Multimodal/MMRAG/audio.mp3?raw=true"
response = requests.get(url)
audio_data = io.BytesIO(response.content)

# Converting to wav and loading
audio_segment = AudioSegment.from_file(audio_data, format="mp3")

# Downsampling to 16000 Hz
 #(this is necessary because a future model requires it to be at 16000Hz)
sampling_rate = 16000
audio_segment = audio_segment.set_frame_rate(sampling_rate)

# Exporting the downsampled audio to a wav file in memory
wav_data = io.BytesIO()
audio_segment.export(wav_data, format="wav")
wav_data.seek(0)  # Back to beginning of IO for reading
wav_file = wave.open(wav_data, 'rb')

# converting the audio data to a numpy array
frames = wav_file.readframes(-1)
audio_waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

# URL of the text file
url = "https://github.com/DanielWarfield1/MLWritingAndResearch/blob/main/Assets/Multimodal/MMRAG/Wiki.txt?raw=true"
response = requests.get(url)
text_data = response.text

# truncating length for compatability with an encoder that accepts a small context
# a different encoder could be used which allows for larger context lengths
text_data = text_data[:300]

## STT
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

#the model that generates text based on speech audio
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-librispeech-asr")
#a processor that gets everything set up
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-librispeech-asr")

#passing through model
inputs = processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

#turning model output into text
audio_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

audio_transcription

## image
query = 'who is my favorite harpist?

from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode the image
inputs = processor(images=image, return_tensors="pt")
image_embeddings = model.get_image_features(**inputs)

# Encode the text
inputs = processor(text=[query, audio_transcription, text_data], return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**inputs)

####### 
import torch
from torch.nn.functional import cosine_similarity

# unpacking individual embeddings
image_embedding = image_embeddings[0]
query_embedding = text_embeddings[0]
audio_embedding = text_embeddings[1]
text_embedding = text_embeddings[2]

# Calculate cosine similarity
cos_sim_query_image = cosine_similarity(query_embedding.unsqueeze(0), image_embedding.unsqueeze(0)).item()
cos_sim_query_audio = cosine_similarity(query_embedding.unsqueeze(0), audio_embedding.unsqueeze(0)).item()
cos_sim_query_text = cosine_similarity(query_embedding.unsqueeze(0), text_embedding.unsqueeze(0)).item()


# putting all the similarities in a list
similarities = [cos_sim_query_image, cos_sim_query_audio, cos_sim_query_text]

result = None
if max(similarities) == cos_sim_query_image:
    #image most similar, augmenting with image
    model = genai.GenerativeModel('gemini-1.5-pro')
    result = model.generate_content([query, Image.open('image.jpeg')])

elif max(similarities) == cos_sim_query_audio:
    #audio most similar, augmenting with audio. Here I'm using the transcript
    #rather than the audio itself
    model = genai.GenerativeModel('gemini-1.5-pro')
    result = model.generate_content([query, 'audio transcript (may have inaccuracies): '+audio_transcription])

elif max(similarities) == cos_sim_query_text:
    #text most similar, augmenting with text
    model = genai.GenerativeModel('gemini-1.5-pro')
    result = model.generate_content([query, text_data])

print(result.text)
