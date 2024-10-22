## From https://medium.com/analytics-vidhya/parler-tts-the-best-open-source-text-to-speech-library-438fb727afce

import pyttsx3

engine = pyttsx3.init()

def set_rate(rate=200):
    engine.setProperty('rate', rate)

def set_volume(volume=1.0):
    engine.setProperty('volume', volume)

def set_gender(gender='male'):
    voices = engine.getProperty('voices')
    
    if gender == 'female':
        engine.setProperty('voice', voices[1].id)  # female voice
    else:
        engine.setProperty('voice', voices[0].id)  # male voice


set_rate(150)        
set_volume(0.9)       
set_gender('female')  

engine.save_to_file("Hello, this is Satyajeet.", 'output_audio.wav')
engine.runAndWait()

------------------------------------------------------------------------------------------
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

text_input = texttospeech.SynthesisInput(text="Hello, I hope you're having a great day!")

voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Wavenet-D",
    ssml_gender=texttospeech.SsmlVoiceGender.MALE,
)


audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.2, 
    pitch=5.0,      
    volume_gain_db=-1.0  
)


response = client.synthesize_speech(
    input=text_input, voice=voice, audio_config=audio_config
)

with open("output.wav", "wb") as out:
    out.write(response.audio_content)
    print("Audio content written to file 'output.wav'")

------------------------------------------------------------------------------------------
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

################
# In Windows
!pip install git+https://github.com/huggingface/parler-tts.git

# In Apple Silicon
!pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

