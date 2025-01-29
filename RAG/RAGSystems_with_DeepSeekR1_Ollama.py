### From https://sebastian-petrus.medium.com/developing-rag-systems-with-deepseek-r1-ollama-f2f561cfda97

ollama run deepseek-r1  # For the 7B model (default)  
ollama run deepseek-r1:1.5b

import streamlit as st  
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama

# Streamlit file uploader  
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")  

if uploaded_file:  
    # Save PDF temporarily  
    with open("temp.pdf", "wb") as f:  
        f.write(uploaded_file.getvalue())  

    # Load PDF text  
    loader = PDFPlumberLoader("temp.pdf")  
    docs = loader.load()

# Split text into semantic chunks  
text_splitter = SemanticChunker(HuggingFaceEmbeddings())  
documents = text_splitter.split_documents(docs)

# Generate embeddings  
embeddings = HuggingFaceEmbeddings()  
vector_store = FAISS.from_documents(documents, embeddings)  

# Connect retriever  
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks

llm = Ollama(model="deepseek-r1:1.5b")  # Our 1.5B parameter model  

# Craft the prompt template  
prompt = """  
1. Use ONLY the context below.  
2. If unsure, say "I don’t know".  
3. Keep answers under 4 sentences.  

Context: {context}  

Question: {question}  

Answer:  
"""  
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# Chain 1: Generate answers  
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)  

# Chain 2: Combine document chunks  
document_prompt = PromptTemplate(  
    template="Context:\ncontent:{page_content}\nsource:{source}",  
    input_variables=["page_content", "source"]  
)  

# Final RAG pipeline  
qa = RetrievalQA(  
    combine_documents_chain=StuffDocumentsChain(  
        llm_chain=llm_chain,  
        document_prompt=document_prompt  
    ),  
    retriever=retriever  
)

# Streamlit UI  
user_input = st.text_input("Ask your PDF a question:")  

if user_input:  
    with st.spinner("Thinking..."):  
        response = qa(user_input)["result"]  
        st.write(response)
