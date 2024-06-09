## https://generativeai.pub/summarising-text-and-converting-to-speech-using-huggingface-and-langchain-concept-tutorial-706ed89455a7

from datasets import load_dataset
import torch
import soundfile as sf
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

def chunks_and_document(txt):
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    return docs

def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])  
    llm = CTransformers(
        model=r"llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        max_new_tokens=1000,
        temperature=0.5
    )
    return llm

def chains_and_response(docs):
    llm = load_llm()
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

txt_input = """ Why AI Agents
I have to say that all the talk about AI agents has been in the background for me, so far. First of all, because I have this idea that I want to do things only with Open-Source models; and secondly because I was looking for small but accurate models to move into the AI Agents world. Remember that I don’t have a GPU, so for me, the choice and the size of the model is always a priority.

But… what the hell are these agents?
I am a Process Control Industrial Automation engineer, so it comes easy to me to explain what Agents are in this way: agents are the driving force behind decision-making processes. They are

computer programs or systems designed to interact with their environment, make choices, and reach specific objectives.
not directly controlled by humans: these autonomous entities operate independently, enabling flexible problem-solving capabilities.
Agents can be categorized based on their distinct characteristics, such as their responsiveness (reactive) or proactive nature, the stability of their environment (fixed or dynamic), and their involvement in multi-agent systems.

Reactive agents react promptly to environmental stimuli and take actions based on these inputs
Proactive agents proactively plan and act to achieve their goals.
When multiple agents collaborate, they form a multi-agent system, each contributing to a common objective. To ensure effective coordination and communication, these agents must synchronize their actions and interact with one another.

"""
result = []
docs = chunks_and_document(txt_input)
response = chains_and_response(docs)
result.append(response)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text=result, return_tensors="pt")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("speech.wav", speech.numpy(), samplerate=16000)

