### From https://medium.com/data-science-collective/building-a-robust-rag-system-with-langfuse-and-ragas-a-complete-implementation-guide-with-pyhton-64dd73fb0657

!pip install langchain langchain_openai faiss-cpu ragas pypdf langfuse

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langfuse import Langfuse
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI as OpenAILLM
from dotenv import load_dotenv
load_dotenv()

def ask_with_langfuse(query, trace):
    query_generation = trace.generation(
        name="query_execution",
        model="gpt-4o",
        model_parameters={"max_tokens": 256},
        input={"query": query}
    )
    
    try:
        # Execute query
        result = qa({"query": query})
        
        # Extract source documents for logging in a serializable format
        source_docs = []
        for doc in result["source_documents"][:2]:
            # Ensure metadata is serializable
            metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    metadata[key] = value
            
            source_docs.append({
                "content": doc.page_content,
                "metadata": metadata
            })
        
        # Update the generation with the result
        query_generation.end(
            output={"answer": result["result"]},
            metadata={"source_count": len(result["source_documents"])}
        )
        
        return result
    except Exception as e:
        # Log any errors
        query_generation.end(
            error={"message": str(e), "type": type(e).__name__}
        )
        trace.update(status="error")
        raise e

# Initialize Langfuse directly
langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Set LangChain tracing if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Load your PDF file
print("Loading PDF...")
loader = PyPDFLoader("data/documents/mamba model.pdf")
pages = loader.load()

# Create a trace for the entire process
main_trace = langfuse.trace(
    name="rag_pdf_process",
    user_id="user-001",
    metadata={"file": "mamba model.pdf"}
)

document_splitting = main_trace.span(
    name="document_splitting",
    input={"page_count": len(pages)}
)

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_documents(pages)

document_splitting.update(
    output={"chunk_count": len(chunks)}
)
document_splitting.end()

vectorization = main_trace.span(
    name="vectorization",
    input={"chunk_count": len(chunks)}
)

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

vectorization.end()

# Build RAG chain
chain_setup = main_trace.span(name="chain_setup")

llm = OpenAILLM(
    model_name="gpt-4o",
    max_tokens=256,
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

chain_setup.end()

query = "What are the main topics covered in the PDF?"

# Run the query
print("Running query...")
response = ask_with_langfuse(query, main_trace)
print("Answer:", response["result"])

# 🧪 Evaluate using RAGAS
print("Evaluating with RAGAS...")
eval_span = main_trace.span(name="ragas_evaluation")

contexts = [doc.page_content for doc in response["source_documents"][:2]]

# Create a dataset compatible with RAGAS
eval_dataset = Dataset.from_dict({
    "question": [query],
    "answer": [response["result"]],
    "contexts": [contexts],
    "ground_truth": ["Summary of main PDF topics"]
})

# Run evaluation
try:
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    
    # Convert evaluation results to a simple format
    metrics = {}
    
    # Handle the result object based on its string representation
    result_str = str(result)
    print("RAGAS result:", result_str)
    
    # Try to extract values directly if possible
    try:
        # First try accessing as dictionary
        metrics["faithfulness"] = float(result["faithfulness"])
        metrics["answer_relevancy"] = float(result["answer_relevancy"])
    except (TypeError, KeyError):
        # If that fails, try parsing the string representation
        import re
        faithfulness_match = re.search(r"faithfulness[^\d]+([\d\.]+)", result_str)
        relevancy_match = re.search(r"answer_relevancy[^\d]+([\d\.]+)", result_str)
        
        if faithfulness_match:
            metrics["faithfulness"] = float(faithfulness_match.group(1))
        if relevancy_match:
            metrics["answer_relevancy"] = float(relevancy_match.group(1))
    
    # Update the evaluation span with metrics
    eval_span.update(
        output={"metrics": metrics}
    )
    
    print("Evaluation metrics:", metrics)
except Exception as e:
    print(f"RAGAS evaluation error: {e}")
    eval_span.update(
        error={"message": str(e), "type": type(e).__name__}
    )

eval_span.end()

# End the main trace
main_trace.update(status="success")

print("RAG process completed and logged to Langfuse")
