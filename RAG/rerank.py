## From https://towardsdatascience.com/improving-rag-performance-using-rerankers-6adda61b966d
!pip install PyMuPDF
!pip install pytesseract
!pip install langchain

import fitz
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch

def parse_document(document_path: str):
  texts = []
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

  pdf_document = fitz.open(document_path)
  page_numbers = list(range(1, 39))
  for page_number in page_numbers:
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    chunked_texts = text_splitter.split_text(text)
    texts.extend(chunked_texts)
  return texts

def setup_embedding_model():
  tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
  model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
  model.eval()
  model.to("cuda")
  return tokenizer, model

def create_embedding(texts, tokenizer, model):
  encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")
  with torch.no_grad():
    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()

chunked_texts = parse_document("./odyssey_stories.pdf")
embedding_tokenizer, embedding_model = setup_embedding_model()
embeddings = create_embedding(chunked_texts, embedding_tokenizer, embedding_model)

query = "Why was Odysseus stuck with Calypso?"
query_embedding = create_embedding([query], embedding_tokenizer, embedding_model)

### Just embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(query_embedding, embeddings)
similarity = similarity[0]

indexed_numbers = list(enumerate(similarity))
sorted_indexed_numbers = sorted(indexed_numbers, key=lambda x: x[1], reverse=True)
sorted_indices = [index for index, number in sorted_indexed_numbers]

top_k = 10
print(f"Original query: {query} \n")
for i in sorted_indices[:top_k]:
  print(texts[i])
  print("\n")


### Add rerankers
def setup_reranker():
  tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
  model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
  model.eval()
  model.to("cuda")
  return tokenizer, model

def run_reraker(text_pairs, tokenizer, model):
  with torch.no_grad():
      inputs = tokenizer(text_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
      scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
      return scores.tolist()

reranker_tokenize, reranker_model = setup_reranker()

pairs = []
for index in sorted_indices[:top_k]:
  pairs.append([query, texts[index]])

scores = run_reraker(pairs, reranker_tokenize, reranker_model)
paired_list = list(zip(sorted_indices[:top_k], scores))
sorted_paired_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
reranked_indices = [index for index, value in sorted_paired_list]
reranked_values = [value for index, value in sorted_paired_list]

print(f"Original query: {query} \n")

for i in reranked_indices:
  print(texts[i])
  print("\n")
