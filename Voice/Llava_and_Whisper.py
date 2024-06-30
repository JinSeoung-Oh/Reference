## From https://medium.com/@kagglepro.llc/building-an-ai-voice-assistant-with-llava-and-whisper-5ca1c9982e35
## If add some fuction on this code for image-to-text, then we can build image-to-speech gradio app <-- I will do one day

import whisper
import llava
from transformers import AutoTokenizer

# Load models
whisper_model = whisper.load_model("base")
llava_model = llava.LlavaModel.from_pretrained("llava-base")
tokenizer = AutoTokenizer.from_pretrained("llava-base")

# Function to preprocess audio
def preprocess_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    return whisper.pad_or_trim(audio)

# Function to preprocess text
def preprocess_text(text):
    return tokenizer(text, return_tensors="pt")

def generate_response(text):
    inputs = preprocess_text(text)
    outputs = llava_model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def transcribe_audio(audio_path):
    audio = preprocess_audio(audio_path)
    result = whisper_model.transcribe(audio)
    return result["text"]

import gradio as gr

def voice_assistant(audio_path):
    text = transcribe_audio(audio_path)
    response = generate_response(text)
    return response

# Create Gradio interface
interface = gr.Interface(fn=voice_assistant, 
                         inputs=gr.inputs.Audio(source="microphone", type="filepath"),
                         outputs="text")

interface.launch()



####
# Example deployment script for Heroku
heroku create
git add .
git commit -m "Initial commit"
git push heroku main
####
