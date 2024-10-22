### It is not official Implement
### From https://towardsdatascience.com/implementing-anthropics-contextual-retrieval-for-powerful-rag-performance-b85173a65b83
import os
import tiktoken
import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from openai import OpenAI
from constants import INPUT_TOKEN_PRICE, OUTPUT_TOKEN_PRICE
from constants import TEXT_EMBEDDING_MODEL, GPT_MODEL
from constants import CHUNK_OVERLAP, CHUNK_SIZE, TEXT_EMBEDDING_MODEL

import os
from utility.openai_utility import count_tokens
import tiktoken
import json


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

INPUT_TOKEN_PRICE = 0.15/1e6
OUTPUT_TOKEN_PRICE = 0.6/1e6

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(text):
    encoding = tiktoken.encoding_for_model(TEXT_EMBEDDING_MODEL)
    tokens = encoding.encode(text)
    return len(tokens)

def get_embedding(text, model=TEXT_EMBEDDING_MODEL):
 return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

def prompt_gpt(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model=GPT_MODEL,  # Ensure correct model name is used
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens  # Input tokens (in)
    completion_tokens = response.usage.completion_tokens  # Output tokens (out)
    price = calculate_prompt_cost(prompt_tokens, completion_tokens)
    return content, price

def calculate_prompt_cost(input_tokens, output_tokens):
    return INPUT_TOKEN_PRICE * input_tokens + OUTPUT_TOKEN_PRICE * output_tokens

tokenizer = tiktoken.get_encoding("cl100k_base")


def split_text_into_chunks_with_overlap(text):
    tokens = tokenizer.encode(text)  # Tokenize the input text
    chunks = []
    # Loop through the tokens, creating chunks with overlap
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]  # Include overlap by adjusting start point
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


def load_all_chunks_from_folder(folder_path):
    chunks = []
    for chunk_filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, chunk_filename), "r") as f:
            data = json.load(f)
            chunks.append(data)
    return chunks

########## Creating chunks
import os
from jinja2 import Environment, FileSystemLoader
from tqdm.auto import tqdm
from utility.vector_database_utility import split_text_into_chunks_with_overlap
from utility.openai_utility import prompt_gpt, get_embedding
import json
import streamlit as st

DOCUMENT_FILEPATHS = r"Data\hoyesteretts_dommer\filtered\extracted_info_pages_filtered"
CHUNKS_SAVE_PATH = r"Data\hoyesteretts_dommer\filtered\chunks"
DOCUMENT_TYPE = "hoyesterettsdommer"

# Set up Jinja2 environment
file_loader = FileSystemLoader('./templates')
env = Environment(loader=file_loader)

def get_add_context_prompt(chunk_text, document_text):
 template = env.get_template('create_context_prompt.j2')
 data = {
  'WHOLE_DOCUMENT': document_text,  # Leave blank for default or provide a name
  'CHUNK_CONTENT': chunk_text  # You can set any score here
 }
 output = template.render(data)
 return output

########################
document_filenames = os.listdir(DOCUMENT_FILEPATHS)
for filename in tqdm(document_filenames):
 with open(f"{DOCUMENT_FILEPATHS}/{filename}", "r", encoding="utf-8") as f:
  document_text = f.read()
 # now split text into chunks
 print(f"Current tot price: ", tot_price)
 chunks = split_text_into_chunks_with_overlap(document_text)
 for idx, chunk in enumerate(chunks):
  # store the chunk
  chunk_save_filename = f"{filename.split(".")[0]}_{idx}.json"
  chunk_save_path = f"{CHUNKS_SAVE_PATH}/{chunk_save_filename}"
  
  if os.path.exists(chunk_save_path):
   continue

  prompt = get_add_context_prompt(chunk, document_text)
  context, price = prompt_gpt(prompt)
  tot_price += price
  
  chunk_info = {
   "id" : f"{filename}_{int(idx)}",
   "chunk_text" : context + "\n\n" + chunk,
   "chunk_idx" : idx,
   "filename" : filename,
   "document_type": DOCUMENT_TYPE
  }

  with open(chunk_save_path, "w", encoding="utf-8") as f:
   json.dump(chunk_info, f, indent=4)
----------------------------------------------------------------
# go through each chunk, create an embedding for it, and save it the same folder
for chunk_filename in os.listdir(CHUNKS_SAVE_PATH):
 # load chunk
 with open(f"{CHUNKS_SAVE_PATH}/{chunk_filename}", "r", encoding="utf-8") as f:
  chunk_info = json.load(f)
  chunk_text = chunk_info["chunk_text"]
  chunk_text_embedding = get_embedding(chunk_text)
  chunk_info["chunk_embedding"] = chunk_text_embedding
 # save chunk
 with open(f"{CHUNKS_SAVE_PATH}/{chunk_filename}", "w", encoding="utf-8") as f:
  json.dump(chunk_info, f, indent=4)

################################################################
from pinecone import Pinecone
from pinecone_utility import PineconeUtility

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index_name = "lov-avgjorelser"
index = pinecone.Index(index_name)

# pinecone expects list of objects with: [{"id": id, "values": embedding, "metadata", metadata}]

# upload to pinecone
for chunk_filename in os.listdir(CHUNKS_SAVE_PATH):
 # load chunk
 with open(f"{CHUNKS_SAVE_PATH}/{chunk_filename}", "r", encoding="utf-8") as f:
  chunk_info = json.load(f)
  chunk_filename = chunk_info["filename"]
  chunk_idx = chunk_info["chunk_idx"]
  chunk_text = chunk_info["chunk_text"]
  chunk_info["chunk_embedding"] = chunk_text_embedding
  document_type = chunk_info["document_type"]

  metadata = {
   "filename" : chunk_filename,
   "chunk_idx" : chunk_idx,
   "chunk_text" : chunk_text,
   "document_type" : document_type
  }

  data_with_metadata = [{
   "id" : chunk_filename,
   "values" : chunk_text_embedding,
   "metadata" : metadata
  }]
  index.upsert(vectors=data_with_metadata)

##############################################################
# BM25 indexing
!pip install rank_bm25

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from utility.vector_database_utility import load_all_chunks_from_folder

# Download the NLTK tokenizer if you haven't
nltk.download('punkt')
nltk.download('punkt_tab')

CHUNK_PATH = r"Data\hoyesteretts_dommer\filtered\chunks"
chunks = load_all_chunks_from_folder(CHUNK_PATH)

corpus = [chunk["chunk_text"] for chunk in chunks]

# Tokenize each document in the corpus
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus] # should store this somewhere for easy retrieval
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_with_bm25(query: str, corpus: list[str], top_k: int = 2) -> list[str]:
 tokenized_query = word_tokenize(query.lower())
 doc_scores = bm25.get_top_n(tokenized_query, corpus, n=top_k)
 return doc_scores


if __name__ == "__main__":
  query = "fyllekjøring"
  response = retrieve_with_bm25(query, corpus, top_k=10

##############################################################
# Combining BM25 and vector-based chunk retrieval
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from typing import Optional, List
from utility.openai_utility import prompt_gpt, get_embedding

# NOTE this is the file where you made the bm25 index
from bm25 import retrieve_with_bm25

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI()
pinecone = Pinecone(api_key=PINECONE_API_KEY)


class RagAgent:
 def __init__(self, index_name):
  # load pinecone index
  self.index = pinecone.Index(index_name)

 def query_pinecone(self, query, top_k=2, include_metadata: bool = True):
  query_embedding = get_embedding(query)
  query_response = self._query_pinecone_index(query_embedding, top_k=top_k, include_metadata=include_metadata)
  return self._extract_info(query_response)


 def _query_pinecone_index(self, 
  query_embedding: list, top_k: int = 2, include_metadata: bool = True
 ) -> dict[str, any]:
  query_response = self.index.query(
   vector=query_embedding, top_k=top_k, include_metadata=include_metadata, 
  )
  return query_response
 
 def _extract_info(self, response) -> Optional[dict]:
  """extract data from pinecone query response. Returns dict with id, text and chunk idx"""
  if response is None: return None
  res_list = []
  for resp in response["matches"]:
   _id = resp["id"]
   res_list.append(
    {
    "id": _id,
    "chunk_text": resp["metadata"]["chunk_text"],
    "chunk_idx": resp["metadata"]["chunk_idx"],
    })
   
  return res_list
 
 def _combine_chunks(self, chunks_bm25, chunks_vector_db, top_k=20):
  """given output from bm25 and vector database, combine them to only include unique chunks"""
  retrieved_chunks = []
  # assume lists are ordered from most relevant docs to least relevant
  for chunk1, chunk2 in zip(chunks_bm25, chunks_vector_db):
   if chunk1 not in retrieved_chunks:
    retrieved_chunks.append(chunk1)
    if len(retrieved_chunks) >= top_k:
     break
   if chunk2 not in retrieved_chunks:
    retrieved_chunks.append(chunk2)
    if len(retrieved_chunks) >= top_k:
     break
  return retrieved_chunks

 def run_bm25_rag(self, query, top_k=2):

  chunks_bm25 = retrieve_with_bm25(query, top_k)
  chunks_vector_db = self.query_pinecone(query, top_k)

  combined_chunks = self._combine_chunks(chunks_bm25, chunks_vector_db)

  context = "\n".join([chunk["chunk_text"] for chunk in combined_chunks])
  full_prompt = f"Given the following emails {context} what is the answer to the question: {query}"
  response, _ = prompt_gpt(full_prompt)
  return response 

-------------------------------------------------------------------------
from rag_agent import RagAgent
index_name = "lov-avgjorelser"
rag_agent = RagAgent(index_name)
query = "Hva er straffen for fyllekjøring?"
rag_agent.run_bm25_rag(query, top_k=20)
