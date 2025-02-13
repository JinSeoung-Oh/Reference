### From https://medium.com/@mauryaanoop3/building-intelligent-ai-agents-with-deepseek-r1-and-smolagents-e0de20566bf0

# Step 1: Load and Process PDFs
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

def load_and_process_pdfs(data_dir: str):
    """
    Load PDFs from a directory and split them into manageable chunks.

    Args:
        data_dir (str): The directory containing PDF files.

    Returns:
        list: A list of document chunks.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory: str):
    """
    Create and persist a Chroma vector store from document chunks.

    Args:
        chunks (list): The list of document chunks.
        persist_directory (str): Directory to persist the vector store.

    Returns:
        Chroma: The initialized Chroma vector store.
    """
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Creating new vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def main():
    data_dir = "data"
    db_dir = "chroma_db"
    
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()

---------------------------------------------------------------------------------------
# Step 2: Implementing the Reasoning Agent

from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from langchain_chroma import Chroma
import os

reasoning_model_id = "deepseek-r1:7b"

def get_model(model_id):
    """
    Initialize an AI model based on the specified model ID.
    
    Args:
        model_id (str): The ID of the AI model to use.
    
    Returns:
        OpenAIServerModel: The initialized AI model.
    """
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",
        api_key="ollama"
    )

# Create the reasoner for better RAG
reasoning_model = get_model(reasoning_model_id)
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)

# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)
db_dir = "chroma_db"
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

--------------------------------------------------------------------------------------------
# Step 3: Implementing Retrieval-Augmented Generation (RAG)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response.

    Args:
        user_query: The user's question to query the vector database with.

    Returns:
        str: A concise and specific answer to the user's question.
    """
    # Search for relevant documents
    docs = vectordb.similarity_search(user_query, k=3)
    
    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create prompt with context
    prompt = f"""Based on the following context, answer the user's question concisely and specifically.
    If the information is insufficient, suggest a better query for further RAG retrieval.

Context:
{context}

Question: {user_query}

Answer:"""
    
    # Get response from reasoning model
    response = reasoner.run(prompt, reset=False)
    return response

--------------------------------------------------------------------
# Step 4: Implementing the Primary AI Agent

# Create the primary agent to direct the conversation
tool_model = get_model("llama3.2")
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=tool_model, add_base_tools=False, max_steps=3)

def main():
    """
    Launch the AI agent with a web interface using Gradio.
    """
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    main()
