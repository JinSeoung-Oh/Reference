"""
from https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

1. Fixed Size Chunking
   This is the most common and straightforward approach to chunking
   we simply decide the number of tokens in our chunk and, optionally, 
   whether there should be any overlap between them

2. Recursive Chunking
   Recursive chunking divides the input text into smaller chunks in a hierarchical 
   and iterative manner using a set of separators

3. Document Specific Chunking
   It takes into consideration the structure of the document
   Instead of using a set number of characters or recursive process 
   it creates chunks that align with the logical sections of the document like paragraphs or sub sections.

4. Sematic Chunking
   Semantic Chunking considers the relationships within the text. 
   It divides the text into meaningful, semantically complete chunks

5. Agentic Chunk
   The hypothesis here is to process documents in a fashion that humans would do.
   1. We start at the top of the document, treating the first part as a chunk.
   2. We continue down the document, deciding if a new sentence or piece of information 
      belongs with the first chunk or should start a new one
   3. We keep this up until we reach the end of the document.

Semantic chunking involves taking the embeddings of every sentence in the document, 
comparing the similarity of all sentences with each other, 
and then grouping sentences with the most similar embeddings together.
By focusing on the text’s meaning and context,
Semantic Chunking significantly enhances the quality of retrieval. 
It’s a top-notch choice when maintaining the semantic integrity of the text is vital.
   
"""
!pip install -qU langchain_experimental langchain_openai langchain_community langchain ragas chromadb langchain-groq fastembed pypdf openai

langchain==0.1.16

langchain-community==0.0.34

langchain-core==0.1.45

langchain-experimental==0.0.57

langchain-groq==0.1.2

langchain-openai==0.1.3

langchain-text-splitters==0.0.1

langcodes==3.3.0

langsmith==0.1.49

chromadb==0.4.24

ragas==0.1.7

fastembed==0.2.6

### Download dataset

! wget "https://arxiv.org/pdf/1810.04805.pdf"
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyPDFLoader("1810.04805.pdf")
documents = loader.load()
print(len(documents))

### Perform Native Chunking(RecursiveCharacterTextSplitting)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
naive_chunks = text_splitter.split_documents(documents)
for chunk in naive_chunks[10:15]:
  print(chunk.page_content+ "\n")


### Instantiate Embedding Model
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

## Setup the API Key for LLM
from google.colab import userdata
from groq import Groq
from langchain_groq import ChatGroq

groq_api_key = userdata.get("GROQ_API_KEY")

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

for semantic_chunk in semantic_chunks:
  if "Effect of Pre-training Tasks" in semantic_chunk.page_content:
    print(semantic_chunk.page_content)
    print(len(semantic_chunk.page_content))

### Below line for RAG with Chroma
from langchain_community.vectorstores import Chroma
semantic_chunk_vectorstore = Chroma.from_documents(semantic_chunks, embedding=embed_model)

semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k" : 1})
semantic_chunk_retriever.invoke("Describe the Feature-based Approach with BERT?")


from langchain_core.prompts import ChatPromptTemplate

rag_template = """\
Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

User's Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

chat_model = ChatGroq(temperature=0,
                      model_name="mixtral-8x7b-32768",
                      api_key=userdata.get("GROQ_API_KEY"),)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

semantic_rag_chain = (
    {"context" : semantic_chunk_retriever, "question" : RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

semantic_rag_chain.invoke("What is the purpose of Ablation Studies?")

### Implement a RAG pipeline using Naive Chunking Strategy
# naive_chunks == Top k
naive_chunk_vectorstore = Chroma.from_documents(naive_chunks, embedding=embed_model)
naive_chunk_retriever = naive_chunk_vectorstore.as_retriever(search_kwargs={"k" : 5})
naive_rag_chain = (
    {"context" : naive_chunk_retriever, "question" : RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

########################## Ragas Assessment for Semantic Chunker ############################
synthetic_data_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
#
synthetic_data_chunks = synthetic_data_splitter.create_documents([d.page_content for d in documents])
questions = []
ground_truths_semantic = []
contexts = []
answers = []

question_prompt = """\
You are a teacher preparing a test. Please create a question that can be answered by referencing the following context.

Context:
{context}
"""

question_prompt = ChatPromptTemplate.from_template(question_prompt)

ground_truth_prompt = """\
Use the following context and question to answer this question using *only* the provided context.

Question:
{question}

Context:
{context}
"""

ground_truth_prompt = ChatPromptTemplate.from_template(ground_truth_prompt)

question_chain = question_prompt | chat_model | StrOutputParser()
ground_truth_chain = ground_truth_prompt | chat_model | StrOutputParser()

for chunk in synthetic_data_chunks[10:20]:
  questions.append(question_chain.invoke({"context" : chunk.page_content}))
  contexts.append([chunk.page_content])
  ground_truths_semantic.append(ground_truth_chain.invoke({"question" : questions[-1], "context" : contexts[-1]}))
  answers.append(semantic_rag_chain.invoke(questions[-1]))

from datasets import load_dataset, Dataset

qagc_list = []

for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths_semantic):
  qagc_list.append({
      "question" : question,
      "answer" : answer,
      "contexts" : context,
      "ground_truth" : ground_truth
  })

eval_dataset = Dataset.from_list(qagc_list)
eval_dataset

###########################RESPONSE###########################
Dataset({
    features: ['question', 'answer', 'contexts', 'ground_truth'],
    num_rows: 10
})

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

#
from ragas import evaluate

result = evaluate(
    eval_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
     llm=chat_model, 
    embeddings=embed_model,
    raise_exceptions=False
)
