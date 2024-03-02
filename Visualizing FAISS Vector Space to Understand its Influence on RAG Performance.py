# From https://ai.gopubby.com/visualizing-faiss-vector-space-to-understand-its-influence-on-rag-performance-14d71c6a4f47

! pip install langchain faiss-cpu sentence-transformers flask-sqlalchemy psutil unstructured pdf2image unstructured_inference pillow_heif opencv-python pikepdf pypdf
! pip install renumics-spotlight
! CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir # This line above involves installing the llama-cpp-python library with Metal support

## Make LoadFVectorize.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# access an online pdf
def load_doc() -> 'List[Document]':
    loader = OnlinePDFLoader("https://support.riverbed.com/bin/support/download?did=7q6behe7hotvnpqd9a03h1dji&version=9.15.0")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# vectorize and commit to disk
def vectorize(embeddings_model) -> 'FAISS':
    docs = load_doc()
    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local("./opdf_index")
   return db

# attempts to load vectorstore from disk
def load_db() -> 'FAISS':
    embeddings_model = HuggingFaceEmbeddings()
    try:
        db = FAISS.load_local("./opdf_index", embeddings_model)
   except Exception as e:
        print(f'Exception: {e}\nNo index on disk, creating new...')
        db = vectorize(embeddings_model)
    return db

## Make main.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
import LoadFVectorize
from renumics import spotlight
import pandas as pd
import numpy as np

# Prompt template 
qa_template = """<|system|>
You are a friendly chatbot who always responds in a precise manner. If answer is 
unknown to you, you will politely say so.
Use the following context to answer the question below:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""

# Create a prompt instance 
QA_PROMPT = PromptTemplate.from_template(qa_template)
# load LLM
llm = LlamaCpp(
    model_path="./models/tinyllama_gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    temperature=0.01,
    max_tokens=2000,
    top_p=1,
    verbose=False,
    n_ctx=2048
)
# vectorize and create a retriever
db = LoadFVectorize.load_db()
faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}, max_tokens_limit=1000)
# Define a QA chain 
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=faiss_retriever,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

query = 'What versions of TLS supported by Client Accelerator 6.3.0?'

result = qa_chain({"query": query})
print(f'--------------\nQ: {query}\nA: {result["result"]}')

visualize_distance(db,query,result["result"])

## visualize_distance
def visualize_distance(db,query,result["result"]):
  vs = db.__dict__.get("docstore")
  index_list = db.__dict__.get("index_to_docstore_id").values()
  doc_cnt = db.index.ntotal
  embeddings_vec = db.index.reconstruct_n()
  doc_list = list() 
  for i,doc_id in enumerate(index_list):
      a_doc = vs.search(doc_id)
      doc_list.append([doc_id,a_doc.metadata.get("source"),a_doc.page_content,embeddings_vec[i]])
  df = pd.DataFrame(doc_list,columns=['id','metadata','document','embedding'])
  # add rows for question and answer
  embeddings_model = HuggingFaceEmbeddings()
  question_df = pd.DataFrame(
      {
          "id": "question",
          "question": question,
          "embedding": [embeddings_model.embed_query(question)],
      })
  answer_df = pd.DataFrame(
     {
           "id": "answer",
          "answer": answer,
          "embedding": [embeddings_model.embed_query(answer)],
     })
  df = pd.concat([question_df, answer_df, df])

  question_embedding = embeddings_model.embed_query(question)
  # add column for vector distance
  df["dist"] = df.apply(                                                                                                                                                                         
      lambda row: np.linalg.norm(
        np.array(row["embedding"]) - question_embedding
      ),axis=1,)
  
  spotlight.show(df)
