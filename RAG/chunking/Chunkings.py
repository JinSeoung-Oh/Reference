## From https://towardsdatascience.com/using-evaluations-to-optimize-a-rag-pipeline-from-chunkings-and-embeddings-to-llms-40e5ed6033b8

# 1. Recursive Character Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
CHUNK_SIZE = 512
chunk_overlap = np.round(CHUNK_SIZE * 0.10, 0)

# The splitter to use to create smaller (child) chunks.
child_text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=CHUNK_SIZE,
   chunk_overlap=chunk_overlap
)

# Child docs directly from raw docs.
sub_docs = child_text_splitter.split_documents(docs)

# Inspect chunk lengths.
print(f"{len(docs)} docs split into {len(sub_docs)} child documents.")
plot_chunk_lengths(sub_docs, 'Recursive Character')

--------------------------------------------------------------------------------
# 2.Small-to-Big Text Splitting
from langchain_milvus import Milvus
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

# The splitter to use to create bigger (parent) chunks.
PARENT_CHUNK_SIZE = 1586
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
)

# Parent docs for inspection.
parent_docs = parent_splitter.split_documents(docs)

# Inspect chunk lengths.
print(f"{len(docs)} docs split into {len(parent_docs)} parent documents.")
plot_chunk_lengths(parent_docs, 'Parent')

# Create vectorstore for vector indexing and retrieval.
vectorstore = Milvus(
    collection_name="MilvusDocs",
    embedding_function=embed_model,
    connection_args={"uri": "./milvus_demo.db"},
    auto_id=True,
    drop_old=True,
)

# Create doc storage for the parent documents.
store = InMemoryStore()

# Create the ParentDocumentRetriever.
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, 
    docstore=store, 
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# When we add documents two things will happen:
# Parent chunks - docs split into large chunks.
# Child chunks - docs split into into smaller chunks.
# Relationship between parent and child is kept.
retriever.add_documents(docs, ids=None)

# The vector store alone will retrieve small chunks:
child_results = vectorstore.similarity_search(
    SAMPLE_QUESTION,
    k=2)

print(f"Question: {SAMPLE_QUESTION}")
for i, child_result in enumerate(child_results):
    context = child_result.page_content
    print(f"Result #{i+1}, len: {len(context)}")
    print(f"chunk: {context}")
    pprint.pprint(f"metadata: {child_result.metadata}")

# Whereas the doc retriever will return the larger parent document:
parent_results = retriever.get_relevant_documents(SAMPLE_QUESTION)

# Print the retrieved chunk and metadata.
print(f"Num parent results: {len(parent_results)}")
for i, parent_result in enumerate(parent_results):
    print(f"Result #{i+1}, len: {len(parent_result.page_content)}")
    print(f"chunk: {parent_result.page_content}")
    pprint.pprint(f"metadata: {parent_result.metadata}")

--------------------------------------------------------------------------------
# 3. Semantic Text Splitting
from langchain_experimental.text_splitter import SemanticChunker

semantic_docs = []
for doc in docs:

   # Initialize the SemanticChunker with the embedding model.
   text_splitter = SemanticChunker(embed_model)
   semantic_list = text_splitter.create_documents([cleaned_content])

   # Append the list of semantic chunks to semantic_docs.
   semantic_docs.extend(semantic_list)

# Inspect chunk lengths
print(f"Created {len(semantic_docs)} semantic documents from {len(docs)}.")
plot_chunk_lengths(semantic_docs, 'Semantic')

# Create vectorstore for vector index and retrieval.
vectorstore = Milvus.from_documents(
    collection_name="MilvusDocs",
    documents=semantic_docs,
    embedding=embed_model,
    connection_args={"uri": "./milvus_demo.db"},
    drop_old=True,
)

# Retrieve semantic chunks.
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
semantic_results = semantic_retriever.invoke(SAMPLE_QUESTION)

print(len(semantic_results))
print(f"Question: {SAMPLE_QUESTION}")

# Print the retrieved chunk and metadata.
for i, semantic_result in enumerate(semantic_results):
    context = semantic_result.page_content
    print(f"Result #{i+1}, len: {len(context)}")
    print(f"chunk: {context}")
    pprint.pprint(f"metadata: {semantic_result.metadata}")

--------------------------------------------------------------------------------
## 4. Evaluating the Chunking Methods
import pandas as pd
import ragas, datasets
# Libraries to customize ragas critic model.
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOllama
# Libraries to customize ragas embedding model.
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import the evaluation metrics.
from ragas.metrics import (
    context_recall, 
    context_precision, 
    )

# Read ground truth answers from a CSV file.
eval_df = pd.read_csv(file_path, header=0, skip_blank_lines=True)

##########################################
# Set the evaluation type.
EVALUATE_WHAT = 'CONTEXTS'
##########################################

# Set the columns to evaluate.
if EVALUATE_WHAT == 'CONTEXTS':
    cols_to_evaluate=\
    ['recursive_context_512_k_2', 'parent_context_1536_k1',
     'semantic_context_k_1', 'semantic_context_k_2_summary']

# Set the metrics to evaluate.
if EVALUATE_WHAT == 'CONTEXTS':
    eval_metrics=[
        context_recall, 
        context_precision,
        ]
    metrics = ['context_recall', 'context_precision']
    
# Change the default llm-as-critic model to local llama3.
LLM_NAME = 'llama3'
ragas_llm = LangchainLLMWrapper(langchain_llm=ChatOllama(model=LLM_NAME))

# Change the default embeddings models to HuggingFace models.
EMB_NAME = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
lc_embed_model = HuggingFaceEmbeddings(
    model_name=EMB_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
ragas_emb = LangchainEmbeddingsWrapper(embeddings=lc_embed_model)

# Change embeddings and critic models for each metric.
for metric in metrics:
    globals()[metric].llm = ragas_llm
    globals()[metric].embeddings = ragas_emb

# Execute the evaluation.
print(f"Evaluating {EVALUATE_WHAT} using {eval_df.shape[0]} eval questions:")
ragas_result, scores = _eval_ragas.evaluate_ragas_model(
    eval_df, 
    eval_metrics, 
    what_to_evaluate=EVALUATE_WHAT,
    cols_to_evaluate=cols_to_evaluate)



