## From https://medium.com/@datadrifters/llama-3-powered-voice-assistant-integrating-local-rag-with-qdrant-whisper-and-langchain-b4d075b00ac5

"""
pip3 install --no-deps torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip3 install openai
pip3 install -q transformers==4.33.0 
pip3 install -q accelerate==0.22.0 
pip3 install -q einops==0.6.1 
pip3 install -q langchain==0.0.300 
pip3 install -q xformers==0.0.21
pip3 install -q bitsandbytes==0.41.1 
pip3 install -q sentence_transformers==2.2.2
pip3 install arxiv
pip3 install -q ipykernel jupyter
pip3 install -q --upgrade huggingface_hub

pip3 install unstructured
pip3 install "unstructured[pdf]"
apt-get install -y poppler-utils
pip3 install pytesseract
apt-get install -y tesseract-ocr
pip3 install --upgrade qdrant-client
pip3 install WhisperSpeech
"""

import os
import sys
import arxiv
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader,DirectoryLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from pathlib import Path
from openai import OpenAI
from IPython.display import Audio, display
from whisperspeech.pipeline import Pipeline

dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
   os.makedirs(dirpath)
   
search = arxiv.Search(
  query = "LLM", # your query length is limited by ARXIV_MAX_QUERY_LENGTH which is 300 characters
  max_results = 10,
  sort_by = arxiv.SortCriterion.LastUpdatedDate, # you can also use SubmittedDate or Relevance
  sort_order = arxiv.SortOrder.Descending
)

for result in search.results():
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            break
        except FileNotFoundError:
            print("File not found")
            break
        except HTTPError:
            print("Forbidden")
            break
        except ConnectionResetError as e:
            print("Connection reset by peer")
            time.sleep(5)

papers = []
loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
papers = loader.load()
print("Total number of pages loaded:", len(papers)) # Total number of pages loaded: 410

# This merges all papes from all papers into single text block for chunking
full_text = ''
for paper in papers:
    full_text = full_text + paper.page_content
    
full_text = " ".join(l for l in full_text.splitlines() if l)
print(len(full_text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50
)

paper_chunks = text_splitter.create_documents([full_text])

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto",)

llm = HuggingFacePipeline(pipeline=query_pipeline)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

# try to access the sentence transformers from HuggingFace: https://huggingface.co/api/models/sentence-transformers/all-mpnet-base-v2
try:
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
except Exception as ex:
    print("Exception: ", ex)
    # # alternatively, we will access the embeddings models locally
    # local_model_path = "/kaggle/input/sentence-transformers/minilm-l6-v2/all-MiniLM-L6-v2"
    # print(f"Use alternative (local) model: {local_model_path}\n")
    # embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs=model_kwargs)

vectordb = Qdrant.from_documents(
    paper_chunks,
    embeddings,
    path="Qdrant_Persist",
    collection_name="voice_assistant_documents",
)

from qdrant_client import QdrantClient

client = QdrantClient(path = "Qdrant_Persist")

vectordb = Qdrant(
    client=client,
    collection_name="voice_assistant_documents",
    embeddings=embeddings,
)

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)


from IPython.display import display, Markdown

def colorize_text(text):
    for word, color in zip(["Reasoning", "Question", "Answer", "Total time"], ["blue", "red", "green", "magenta"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

def test_rag(qa, query):

    time_start = time()
    response = qa.run(query)
    time_end = time()
    total_time = f"{round(time_end-time_start, 3)} sec."

    full_response =  f"Question: {query}\nAnswer: {response}\nTotal time: {total_time}"
    display(Markdown(colorize_text(full_response)))
    return response


pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

query = "How LLMs can be used to understand and interact with the complex 3D world"
aud = test_rag(qa, query)

pipe.generate_to_notebook(f"{aud}")













