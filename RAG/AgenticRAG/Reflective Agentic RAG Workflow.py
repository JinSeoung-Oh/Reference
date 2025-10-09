### From https://nayakpplaban.medium.com/build-a-reflective-agentic-rag-workflow-using-langgraph-typesense-tavily-ollama-and-cohere-c9a7b0aca667

uv pip install streamlit pandas langchain langchain-community langchain-groq 
langchain-ollama typesense pypdf python-dotenv langextract

import streamlit as st
import pandas as pd
import time
import os
import tempfile
from typing import List, Dict, Any, TypedDict, Annotated
import hashlib
from datetime import datetime
import json
import uuid
from operator import add
import re

# Import core libraries that are always needed
from pydantic import BaseModel, Field

# Try to import langchain core components needed for type definitions
try:
    from langchain_core.messages import BaseMessage
except ImportError:
    # Create a dummy BaseMessage if langchain is not available
    BaseMessage = object

# Import necessary libraries (you'll need to install these)
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_ollama.embeddings import OllamaEmbeddings
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import Typesense
    from langchain_core.documents import Document
    from langchain.chains import RetrievalQA
    from langchain.retrievers import  EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.document_compressors import FlashrankRerank
    from langchain_cohere import CohereRerank
    from langchain_tavily import TavilySearch
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    import typesense
    import langextract as lx
    import textwrap
except ImportError as e:
    st.error(f"Missing required libraries. Please install: {e}")

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Hub",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #3498db;
        background-color: #f8f9fa;
    }
    
    .success-box {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .progress-container {
        background: #f1f3f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define Pydantic models for structured output
class crFormat(BaseModel):
    Web_Response: str = Field(..., description="response generated for the question asked based on the information provided.")

# Define state schema for agentic RAG
class AgentState(TypedDict):
    question: str
    context: List[str]
    generation: str
    reflection: str
    messages: Annotated[List[BaseMessage], add]

class DocumentProcessor:
    def __init__(self):
        self.typesense_client = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.chunks = []
        self.collection_name = "documents"
        self.enhanced_docs = []
        self.compression_retriever = None
        self.agentic_rag_graph = None
        self.current_file_collection = None
        
    def initialize_typesense_client(self, typesense_host: str, typesense_port: str, typesense_api_key: str):
        """Initialize Typesense client"""
        try:
            self.typesense_client = typesense.Client({
                'nodes': [{
                    'host': typesense_host,
                    'port': typesense_port,
                    'protocol': 'http'
                }],
                'api_key': typesense_api_key,
                'connection_timeout_seconds': 60
            })
            
            # Test connection
            self.typesense_client.collections.retrieve()
            return True
        except Exception as e:
            st.error(f"Error connecting to Typesense: {str(e)}")
            return False
    
    def initialize_typesense_client_with_protocol(self, typesense_host: str, typesense_port: str, typesense_api_key: str, protocol: str):
        """Initialize Typesense client with custom protocol"""
        try:
            self.typesense_client = typesense.Client({
                'nodes': [{
                    'host': typesense_host,
                    'port': typesense_port,
                    'protocol': protocol
                }],
                'api_key': typesense_api_key,
                'connection_timeout_seconds': 60
            })
            
            # Test connection
            self.typesense_client.collections.retrieve()
            return True
        except Exception as e:
            st.error(f"Error connecting to Typesense: {str(e)}")
            return False
    
    def check_collection_exists(self, collection_name: str):
        """Check if a Typesense collection exists - using reference app logic"""
        try:
            collections = self.typesense_client.collections.retrieve()
            collection_names = [col['name'] for col in collections]
            print(f"Collection_Name:{collection_name}")
            return collection_name in collection_names
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid api token" in error_msg.lower():
                st.error("🔑 **Invalid Typesense API Key!** Please check your API key in the sidebar.")
            elif "404" in error_msg or "not found" in error_msg.lower():
                st.warning("📡 **Typesense server not found.** Please check your host and port settings.")
            else:
                st.error(f"❌ **Typesense connection error:** {error_msg}")
            print(f"Error checking collections: {error_msg}")
            return False
    
    def generate_collection_name_from_file(self, filename: str):
        """Generate a valid Typesense collection name from filename"""
        import re
        # Remove file extension
        name = os.path.splitext(filename)[0]
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it starts with a letter
        if name and name[0].isdigit():
            name = f"doc_{name}"
        # Limit length to 64 characters (Typesense limit)
        name = name[:64]
        # Ensure it's not empty
        if not name:
            name = "document"
        return name.lower()
    
    def get_collection_name_for_files(self, uploaded_files):
        """Generate collection name based on uploaded files"""
        if len(uploaded_files) == 1:
            # Single file - use file name
            return self.generate_collection_name_from_file(uploaded_files[0].name)
        else:
            # Multiple files - create a combined name or use a generic one
            file_names = [self.generate_collection_name_from_file(f.name) for f in uploaded_files[:3]]  # Use first 3 files
            combined_name = "_".join(file_names)
            if len(combined_name) > 60:  # Leave room for potential suffix
                combined_name = combined_name[:60]
            return combined_name or "multi_documents"
    
    def create_vectorstore_from_documents(self, chunks, embeddings, typesense_host, typesense_port, typesense_api_key, typesense_protocol):
        """Create vectorstore using from_documents - reference app approach"""
        try:
            collection_exists = self.check_collection_exists(self.collection_name)
            
            if not collection_exists:
                st.info(f"Creating new Typesense collection '{self.collection_name}'")
                docsearch = Typesense.from_documents(
                    chunks,
                    embeddings,
                    typesense_client_params={
                        "host": typesense_host,
                        "port": typesense_port,
                        "protocol": typesense_protocol,
                        "typesense_api_key": typesense_api_key,
                        "typesense_collection_name": self.collection_name,
                    },
                )
            else:
                st.success(f"Using existing Typesense collection '{self.collection_name}'")
                docsearch = Typesense(
                    embedding=embeddings,
                    typesense_client=self.typesense_client,
                    typesense_collection_name=self.collection_name
                )
            
            return docsearch
            
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return None
    
    def load_document(self, file_path: str, file_type: str):
        """Load document based on file type"""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path)
            elif file_type == "csv":
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return None
    
    def chunk_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def extract_metadata(self, documents: List, progress_callback=None):
        """Extract metadata from documents using LangExtract"""
        try:
            # Define the extraction schema for metadata
            extraction_class = "document_metadata"
            attributes = ["title", "author", "date", "intent"]
            
            # Provide few-shot examples for accurate extraction
            examples = [
                lx.data.ExampleData(
                    text="The annual report was authored by Jane Smith in January 2023.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=extraction_class,
                            extraction_text="The annual report was authored by Jane Smith in January 2023.",
                            attributes={"title": "annual report", "author": "Jane Smith", "date": "January 2023", "intent": "The annual report representing the company annual turnover was presented by Jane Smith"}
                        )
                    ]
                ),
                lx.data.ExampleData(
                    text="Project timeline document by John Doe on 2022-12-15.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=extraction_class,
                            extraction_text="Project timeline document by John Doe on 2022-12-15.",
                            attributes={"title": "Project timeline document", "author": "John Doe", "date": "2022-12-15", "intent": "This document explains the project timelines in detail"}
                        )
                    ]
                ),
            ]
            
            # Define a concise prompt
            prompt = textwrap.dedent("""\
            Extract document metadata such as title, author, date, and intent.
            Use exact text for extractions. Do not paraphrase or overlap entities.
            Provide meaningful attributes for each entity to add context.""")
            
            # Extract metadata from all documents
            doc_texts = [doc.page_content for doc in documents]
            results = []
            total_docs = len(doc_texts)
            
            for i, input_text in enumerate(doc_texts):
                try:
                    # Run the extraction
                    result = lx.extract(
                        text_or_documents=input_text,
                        prompt_description=prompt,
                        language_model_type=lx.inference.OllamaLanguageModel,
                        examples=examples,
                        model_id="llama3.2",
                        temperature=0.3,
                        model_url="http://localhost:11434"
                    )
                    
                    if result.extractions:
                        metadata = result.extractions[0].attributes
                        results.append(metadata)
                    else:
                        results.append(None)
                        
                    if progress_callback:
                        progress = (i + 1) / total_docs
                        progress_callback(progress)
                        
                except Exception as e:
                    st.warning(f"Metadata extraction failed for document {i+1}: {str(e)}")
                    results.append(None)
                    
                    if progress_callback:
                        progress = (i + 1) / total_docs
                        progress_callback(progress)
            
            # Enhance documents with extracted metadata
            enhanced_docs = []
            for i, doc in enumerate(documents):
                if i < len(results) and results[i] is not None:
                    metadata = results[i]  # This is a dictionary of extracted attributes
                    doc.metadata.update(metadata)
                enhanced_docs.append(doc)
            
            return results
            
        except Exception as e:
            st.error(f"Error in metadata extraction: {str(e)}")
            return []

    
    def create_typesense_vectorstore(self, chunks: List, typesense_host: str, typesense_port: str, typesense_api_key: str, protocol: str = "https"):
        """Create Typesense vectorstore - reference app approach"""
        try:
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            return self.create_vectorstore_from_documents(chunks, embeddings, typesense_host, typesense_port, typesense_api_key, protocol)
        except Exception as e:
            st.error(f"Error creating Typesense vectorstore: {str(e)}")
            return None
    
    def initialize_docsearch_from_existing(self, collection_name: str = "RAG", typesense_host: str = None, typesense_port: str = None, typesense_api_key: str = None, protocol: str = "https"):
        """Initialize DocSearch from existing collection"""
        try:
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            
            # Create Typesense client
            typesense_client = typesense.Client({
                'nodes': [{
                    'host': typesense_host,
                    'port': typesense_port,
                    'protocol': protocol
                }],
                'api_key': typesense_api_key,
                'connection_timeout_seconds': 60
            })
            
            # Initialize DocSearch from existing collection
            docsearch = Typesense(
                embedding=embeddings,
                typesense_client=typesense_client,
                typesense_collection_name=collection_name,
            )
            
            return docsearch
        except Exception as e:
            st.error(f"Error initializing DocSearch from existing collection: {str(e)}")
            return None
    
    def extract_metadata_with_langextract(self, documents: List, progress_callback=None):
        """Extract metadata from documents using LangExtract"""
        try:
            # Define the extraction schema for metadata
            extraction_class = "document_metadata"
            attributes = ["title", "author", "date", "intent"]
            
            # Provide few-shot examples for accurate extraction
            examples = [
                lx.data.ExampleData(
                    text="The annual report was authored by Jane Smith in January 2023.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=extraction_class,
                            extraction_text="The annual report was authored by Jane Smith in January 2023.",
                            attributes={
                                "title": "annual report", 
                                "author": "Jane Smith", 
                                "date": "January 2023", 
                                "intent": "The annual report representing the company annual turnover was presented by Jane Smith"
                            }
                        )
                    ]
                ),
                lx.data.ExampleData(
                    text="Project timeline document by John Doe on 2022-12-15.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=extraction_class,
                            extraction_text="Project timeline document by John Doe on 2022-12-15.",
                            attributes={
                                "title": "Project timeline document", 
                                "author": "John Doe", 
                                "date": "2022-12-15", 
                                "intent": "This document explains the project timelines in detail"
                            }
                        )
                    ]
                )
            ]
            
            # Define a concise prompt
            prompt = textwrap.dedent("""\
            Extract document metadata such as title, author, date, and intent.
            Use exact text for extractions. Do not paraphrase or overlap entities.
            Provide meaningful attributes for each entity to add context.""")
            
            # Extract metadata from all documents in a batch
            doc_texts = [doc.page_content for doc in documents]
            results = []
            
            total_docs = len(doc_texts)
            for i, input_text in enumerate(doc_texts):
                if progress_callback:
                    progress_callback(i / total_docs)
                
                try:
                    # Run the extraction
                    result = lx.extract(
                        text_or_documents=input_text,
                        prompt_description=prompt,
                        language_model_type=lx.inference.OllamaLanguageModel,
                        examples=examples,
                        model_id="llama3.2",
                        temperature=0.3,
                        model_url="http://localhost:11434"
                    )
                    if result.extractions:
                        results.append(result.extractions[0].attributes)
                    else:
                        results.append(None)
                        
                except Exception as e:
                    st.warning(f"Metadata extraction failed for document {i+1}: {str(e)}")
                    results.append(None)
            
            # Enhance documents with extracted metadata
            enhanced_docs = []
            for i, doc in enumerate(documents):
                if i < len(results) and results[i] is not None:
                    metadata = results[i]  # This is a dictionary of extracted attributes
                    doc.metadata.update(metadata)
                enhanced_docs.append(doc)
            
            self.enhanced_docs = enhanced_docs
            return enhanced_docs
            
        except Exception as e:
            st.error(f"Error in metadata extraction: {str(e)}")
            return documents  # Return original documents if extraction fails
    
    def setup_enhanced_retrieval(self, chunks: List, docsearch, cohere_api_key: str = None, use_bm25: bool = False):
        """Setup enhanced retrieval with optional BM25, ensemble, and reranking"""
        try:
            # Initialize vector store retriever
            vector_retriever = docsearch.as_retriever(search_kwargs={"k": 10})
            
            # Setup ensemble retriever (with or without BM25)
            if use_bm25:
                try:
                    # Initialize BM25 retriever for keyword search
                    bm25_retriever = BM25Retriever.from_documents(chunks)
                    bm25_retriever.k = 10
                    
                    # Combine retrievers using EnsembleRetriever
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        weights=[0.6, 0.4]  # Weighted fusion
                    )
                    st.info("✅ Using Vector + BM25 ensemble retrieval")
                except Exception as e:
                    st.warning(f"BM25 setup failed, using vector-only retrieval: {str(e)}")
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever],
                        weights=[1.0]
                    )
            else:
                # Vector-only ensemble
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever],
                    weights=[1.0]
                )
                st.info("✅ Using vector-only retrieval")
            
            # Add reranker for contextual compression
            if cohere_api_key:
                try:
                    reranker = CohereRerank(model="rerank-english-v3.0", cohere_api_key=cohere_api_key)
                    st.info("✅ Using Cohere reranker")
                except Exception as e:
                    st.warning(f"Cohere reranker failed: {str(e)}")
                    reranker = FlashrankRerank()
                    st.info("✅ Using Flashrank reranker (Cohere failed)")
            else:
                reranker = FlashrankRerank()
                st.info("✅ Using Flashrank reranker")
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=ensemble_retriever
            )
            
            self.compression_retriever = compression_retriever
            return compression_retriever
            
        except Exception as e:
            st.error(f"Error setting up enhanced retrieval: {str(e)}")
            # Fallback to simple retriever
            simple_retriever = docsearch.as_retriever()
            self.compression_retriever = simple_retriever
            return simple_retriever
    
    def setup_agentic_rag_graph(self, groq_api_key: str, tavily_api_key: str = None):
        """Setup the streamlined agentic RAG workflow using LangGraph"""
        try:
            # Initialize components
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=4028, api_key=groq_api_key)
            parser = JsonOutputParser(pydantic_object=crFormat)
            
            # Initialize web search tool if API key provided
            web_tool = None
            if tavily_api_key:
                try:
                    # Set environment variable for Tavily
                    import os
                    os.environ['TAVILY_API_KEY'] = tavily_api_key
                    web_tool = TavilySearch(max_results=5, topic="general")
                except Exception as e:
                    st.warning(f"⚠️ Tavily search not available: {str(e)}")
            
            # Define retrieval node - use compression retriever if available, otherwise vector retriever
            def retrieve_node(state: AgentState):
                question = state["question"]
                try:
                    if hasattr(self, 'compression_retriever') and self.compression_retriever:
                        # Use the enhanced retrieval system
                        retrieved_docs = self.compression_retriever.invoke(question)
                        context = [doc.page_content for doc in retrieved_docs]
                    else:
                        # Fallback to vector retriever
                        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
                        retrieved_docs = vector_retriever.invoke(question)
                        context = [doc.page_content for doc in retrieved_docs]
                    
                    return {"context": context}
                except Exception as e:
                    error_msg = str(e)
                    print(f"Retrieval error: {error_msg}")
                    if "401" in error_msg or "invalid api token" in error_msg.lower():
                        return {"context": ["❌ Authentication failed: Invalid Typesense API key. Please update your API key in the sidebar and try again."]}
                    elif "404" in error_msg:
                        return {"context": ["❌ Collection not found. Please check if the collection exists or recreate it."]}
                    else:
                        return {"context": [f"❌ Unable to retrieve documents: {error_msg[:100]}..."]}
                
            
            # Define generation node
            def generate_node(state: AgentState):
                context_str = "\n\n".join(state["context"]) if state["context"] else "No context available."
                prompt = f"""
Answer the question based on the context below. 
If unsure, say "I cannot answer based on the available context."

Context: {context_str}

Question: {state["question"]}

Answer:
"""
                response = llm.invoke(prompt)
                print(f"Response Generated: {response.content}")
                return {"generation": response.content, "messages": [response.content]}
            
            # Define self-reflection node (agentic critique)
            def reflect_node(state: AgentState):
                if not web_tool:
                    return {"reflection": "Web search not available", "messages": ["I cannot answer based on the available context."]}
                
                try:
                    print("---------------------------AGENT REFLECTION -------------------------------------")
                    response = web_tool.invoke({"query": state['question']})
                    context = response.get('results', [])
                    print(context)
                    
                    # Create reflection prompt
                    reflection_prompt = ChatPromptTemplate.from_template("""
Based on the Query and Content provided. Please generate a detailed verbose response. 
Stick to the CONTENT provided. Do not USE YOUR KNOWLEDGE.

QUERY: {query}
CONTENT: {context}

{format_instructions}

The Answer should be clear, detailed and have a professional tone.
Start the response with 'Based on Web Search performed'.
Also Specify the URLs referred.

OUTPUT FORMAT:
{{"Web_Response":"The Answer should be clear,detailed and have a professional tone.Start the response with 'Based on Web Search performed'.Also Specify the URLs referred."}}
""")
                    
                    # Create the chain
                    critique_chain = reflection_prompt | llm | parser
                    
                    # Invoke the chain
                    output = critique_chain.invoke({
                        "query": state['question'],
                        "context": context,
                        "format_instructions": parser.get_format_instructions()
                    })
                    
                    print(f"CRITIQUE: {output}")
                    return {"reflection": response.get('results', []), "messages": [output["Web_Response"]]}
                except Exception as e:
                    print(f"Reflection error: {str(e)}")
                    return {"reflection": "Web search reflection failed", "messages": ["I cannot answer based on the available context."]}
            
            # Helper function to determine if reflection is needed
            def needs_reflection(state: AgentState):
                gen = state['generation'].lower()
                print(f"Generated response: {state['generation']}")
                if "uncertain" in gen or "cannot answer" in gen:
                    return "reflect"
                return END
            
            # Build state graph
            graph_builder = StateGraph(AgentState)
            graph_builder.add_node("retrieve", retrieve_node)
            graph_builder.add_node("generate", generate_node)
            graph_builder.add_node("reflect", reflect_node)
            graph_builder.set_entry_point("retrieve")
            graph_builder.add_edge("retrieve", "generate")
            graph_builder.add_conditional_edges(
                "generate",
                needs_reflection,
                {"reflect": "reflect", "__end__": END}
            )
            
            # Compile graph
            agentic_rag_graph = graph_builder.compile()
            self.agentic_rag_graph = agentic_rag_graph
            
            return agentic_rag_graph
            
        except Exception as e:
            st.error(f"Error setting up agentic RAG graph: {str(e)}")
            return None
    
    def query_agentic_rag(self, question: str):
        """Query the streamlined agentic RAG system"""
        try:
            if not self.agentic_rag_graph:
                st.error("Agentic RAG graph not initialized")
                return None
            
            # Initialize state with required fields
            initial_state = {
                "question": question, 
                "context": [], 
                "generation": "", 
                "reflection": "", 
                "messages": []
            }
            
            # Execute the graph
            result = self.agentic_rag_graph.invoke(initial_state)
            
            # Return the final answer from messages (latest response)
            if result.get("messages") and len(result["messages"]) > 0:
                final_answer = result["messages"][-1]  # Get the latest message
                source_type = "agentic_rag_with_web_search" if result.get("reflection") else "agentic_rag"
                
                return {
                    "answer": final_answer,
                    "source": source_type,
                    "context": result.get("context", [])
                }
            else:
                # Fallback to generation if no messages
                return {
                    "answer": result.get("generation", "No response generated"),
                    "source": "agentic_rag",
                    "context": result.get("context", [])
                }
                
        except Exception as e:
            st.error(f"Error querying agentic RAG: {str(e)}")
            return None
    
    def setup_qa_chain(self, vectorstore, groq_api_key: str):
        """Setup QA chain with Groq LLM"""
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="mixtral-8x7b-32768",
                temperature=0.1
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True
            )
            
            return qa_chain
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'indexing_complete' not in st.session_state:
        st.session_state.indexing_complete = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}

def validate_api_keys(groq_key: str, google_key: str, typesense_api_key: str, langextract_key: str = "") -> Dict[str, bool]:
    """Validate API keys"""
    validation_results = {
        'groq': bool(groq_key and len(groq_key) > 10),
        'google': bool(google_key and len(google_key) > 10),
        'typesense': bool(typesense_api_key and len(typesense_api_key) > 5),
        'langextract': bool(langextract_key and len(langextract_key) > 10) if langextract_key else True  # Optional
    }
    return validation_results

def display_processing_progress(stage: str, progress: float, message: str):
    """Display processing progress with animations"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{stage}**")
        progress_bar = st.progress(progress)
        st.caption(message)
    
    with col2:
        if progress < 1.0:
            st.markdown("🔄 Processing...")
        else:
            st.markdown("✅ Complete!")

# Function to sanitize filenames to valid Typesense collection names:
def sanitize_collection_name(filename: str) -> str:
    # Lowercase, remove extension, replace non-alphanumeric with underscores
    name = filename.lower()
    name = re.sub(r'\.pdf$', '', name)  # remove .pdf extension
    name = re.sub(r'[^a-z0-9]+', '_', name)  # replace invalid chars with underscore
    return name

def process_documents_reference_style(documents, collection_name, typesense_client, typesense_host, typesense_port, typesense_protocol, typesense_api_key, cohere_api_key, groq_api_key, tavily_api_key):
    """Process documents following reference app logic"""
    
    # LangExtract metadata extraction setup
    extraction_class = "document_metadata"
    attributes = ["title", "author", "date", "intent"]
    examples = [
        lx.data.ExampleData(
            text="The annual report was authored by Jane Smith in January 2023.",
            extractions=[
                lx.data.Extraction(
                    extraction_class=extraction_class,
                    extraction_text="The annual report was authored by Jane Smith in January 2023.",
                    attributes={
                        "title": "annual report",
                        "author": "Jane Smith",
                        "date": "January 2023",
                        "intent": "The annual report representing the company annual turnover was presented by Jane Smith"
                    }
                )
            ]
        ),
        lx.data.ExampleData(
            text="Project timeline document by John Doe on 2022-12-15.",
            extractions=[
                lx.data.Extraction(
                    extraction_class=extraction_class,
                    extraction_text="Project timeline document by John Doe on 2022-12-15.",
                    attributes={
                        "title": "Project timeline document",
                        "author": "John Doe",
                        "date": "2022-12-15",
                        "intent": "This document explains the project timelines in detail"
                    }
                )
            ]
        ),
    ]

    prompt = textwrap.dedent("""
    Extract document metadata such as title, author, date, and intent.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.
    """)

    # Extract metadata for each document page
    doc_texts = [doc.page_content for doc in documents]
    results = []
    extraction_errors = []
    with st.spinner("Extracting metadata with LangExtract..."):
        for idx, input_text in enumerate(doc_texts):
            try:
                result = lx.extract(
                    text_or_documents=input_text,
                    prompt_description=prompt,
                    language_model_type=lx.inference.OllamaLanguageModel,
                    examples=examples,
                    model_id="llama3.2",
                    temperature=0.3,
                    model_url="http://localhost:11434"
                )
                print("Extraction successful!")
                print(result)
                results.append(result.extractions[0].attributes)
            except Exception as e:
                extraction_errors.append(f"Page {idx}: {str(e)}")
                results.append({})

    if extraction_errors:
        st.warning(f"Metadata extraction had some errors on {len(extraction_errors)} pages.")
        for err in extraction_errors:
            st.write(err)
    else:
        st.success("Metadata extraction completed successfully.")

    # Enhance documents with extracted metadata
    enhanced_docs = []
    for i, doc in enumerate(documents):
        if i < len(results) and results[i]:
            doc.metadata.update(results[i])
        enhanced_docs.append(doc)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(enhanced_docs)
    st.info(f"Split documents into {len(chunks)} chunks for indexing.")

    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Create vectorstore - reference app style
    st.info(f"Creating new Typesense collection '{collection_name}'")
    docsearch = Typesense.from_documents(
        chunks,
        embeddings,
        typesense_client_params={
            "host": typesense_host,
            "port": typesense_port,
            "protocol": typesense_protocol,
            "typesense_api_key": typesense_api_key,
            "typesense_collection_name": collection_name,
        },
    )

    # Setup retrievers and reranker - reference app style
    setup_retrievers_and_agentic_rag(docsearch, chunks, cohere_api_key, groq_api_key, tavily_api_key, collection_name)

def setup_existing_collection_reference_style(collection_name, typesense_client, cohere_api_key, groq_api_key, tavily_api_key):
    """Setup existing collection following reference app logic"""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    docsearch = Typesense(
        embedding=embeddings,
        typesense_client=typesense_client,
        typesense_collection_name=collection_name
    )
    
    # Setup retrievers and agentic RAG for existing collection
    setup_retrievers_and_agentic_rag(docsearch, [], cohere_api_key, groq_api_key, tavily_api_key, collection_name)

def setup_retrievers_and_agentic_rag(docsearch, chunks, cohere_api_key, groq_api_key, tavily_api_key, collection_name):
    """Setup retrievers and agentic RAG - reference app style"""
    
    # Setup retrievers and reranker
    if cohere_api_key:
        compressor = CohereRerank(model="rerank-english-v3.0", cohere_api_key=cohere_api_key)
    else:
        compressor = FlashrankRerank()
        
    vector_retriever = docsearch.as_retriever(search_kwargs={"k": 10})
    
    # Simple ensemble retriever (vector only, like reference app)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever],
        weights=[1]
    )
    reranker = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    # Initialize Tavily Search tool
    if tavily_api_key:
        try:
            # Set environment variable for Tavily
            import os
            os.environ['TAVILY_API_KEY'] = tavily_api_key
            tool = TavilySearch(max_results=5, topic="general")
        except Exception as e:
            st.warning(f"⚠️ Tavily search initialization failed: {str(e)}")
            tool = None
    else:
        tool = None

    # Define agent state and parser
    class crFormat(BaseModel):
        Web_Response: str = Field(..., description="Generated response based on provided info.")

    parser = JsonOutputParser(pydantic_object=crFormat)

    class AgentState(TypedDict):
        question: str
        context: List[str]
        generation: str
        reflection: str
        messages: Annotated[List[BaseMessage], add]

    # Initialize Groq LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=4028, api_key=groq_api_key)

    # Retrieval node
    def retrieve_node(state: AgentState):
        question = state["question"]
        retrieved_docs = compression_retriever.invoke(question)
        context = [doc.page_content for doc in retrieved_docs]
        return {"context": context}

    # Generation node
    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""
        Answer the question based on the context below. 
        If unsure, say "I cannot answer based on the available context."
        Context: {context_str}
        Question: {state["question"]}
        Answer:
        """
        response = llm.invoke(prompt)
        print(f"Response Generated: {response.content}")
        return {"generation": response.content,"messages":[response.content]}

    # Define self-reflection node (agentic critique)
    def reflect_node(state: AgentState):
        if not tool:
            return {"reflection": "Web search not available", "messages": [state["generation"]]}
            
        print("---------------------------AGENT REFLECTION -------------------------------------")
        response = tool.invoke({"query": state['question']})
        context = response.get('results', [])
        print(context)
        
        # Create reflection prompt
        reflection_prompt = ChatPromptTemplate.from_template("""
        Based on the Query and Content provided. Please generate a detailed verbose response. 
        Stick to the CONTENT provided. Do not USE YOUR KNOWLEDGE.
        QUERY : {query}
        CONTENT: {context}
        
        {format_instructions}
        The Answer should be clear,detailed and have a professional tone.Start the response with 'Based on Web Search performed '.
        Also Specify the URLs referred.
        
        OUTPUT FORMAT:
        {{"Web_Response":"The Answer should be clear,detailed and have a professional tone.Start the response with 'Based on Web Search performed'.Also Specify the URLs referred."}}
        
        """)
        
        # Create the chain
        critique_chain = reflection_prompt | llm | parser
        
        # Invoke the chain
        output = critique_chain.invoke({
            "query": state['question'],
            "context":context,
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"CRITIQUE: {output}")
        return {"reflection": response.get('results', []),"messages":[output["Web_Response"]]}

    # Decide if reflection needed
    def needs_reflection(state: AgentState):
        gen = state['generation'].lower()
        if "uncertain" in gen or "cannot answer" in gen:
            return "reflect"
        return END

    # Build state graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("reflect", reflect_node)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_conditional_edges(
        "generate",
        needs_reflection,
        {"reflect": "reflect", "__end__": END}
    )
    agentic_rag_graph = graph_builder.compile()

    # Store in session state for query interface
    st.session_state.agentic_rag_graph = agentic_rag_graph
    st.session_state.collection_name = collection_name

    # Display query interface
    display_query_interface_reference_style()

def display_query_interface_reference_style():
    """Display query interface - reference app style"""
    st.header("❓ Ask Questions about the Uploaded Documents")
    
    if hasattr(st.session_state, 'collection_name'):
        st.success(f"📚 **Using Collection:** `{st.session_state.collection_name}`")

    query = st.text_input("Enter your question:")

    if query:
        if hasattr(st.session_state, 'agentic_rag_graph'):
            initial_state = {"question": query, "context": [], "generation": "", "reflection": "", "messages": []}
            with st.spinner("Processing your query..."):
                result = st.session_state.agentic_rag_graph.invoke(initial_state)

            st.subheader("Answer:")
            if result.get("messages") and len(result["messages"]) > 0:
                st.write(result["messages"][-1])
            else:
                st.write(result.get("generation", "No response generated"))
        else:
            st.error("Agentic RAG system not initialized")

def main():
    """Main Streamlit app function - Reference app structure"""
    st.set_page_config(page_title="Document Metadata Extractor & RAG QA", layout="wide")
    
    st.title("📄 Document Metadata Extraction & Retrieval-Augmented Generation (RAG)")
    
    # Sidebar configuration - keep API key inputs
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🔑 API Keys")
        
        # API Key inputs
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter your Groq API key",
            help="Required for LLM inference"
        )

        
        typesense_api_key = st.text_input(
            "Typesense API Key",
            value="",
            type="password",
            placeholder="Enter your valid Typesense API key",
            help="⚠️ Required: Get this from your Typesense Cloud dashboard"
        )

        
        cohere_api_key = st.text_input(
            "Cohere API Key",
            type="password",
            placeholder="Enter your Cohere API key",
            help="Optional for enhanced reranking (falls back to Flashrank)"
        )
        
        tavily_api_key = st.text_input(
            "Tavily API Key",
            type="password",
            placeholder="Enter your Tavily API key",
            help="Optional for web search capabilities in agentic RAG"
        )
        
        
        st.info("ℹ️ **Metadata Extraction**: Using local Ollama (llama3.2) - no API key required!")
        
        st.subheader("🔧 Typesense Configuration")
        
        typesense_host = st.text_input(
            "Typesense Host",
            value="pnfcivy9jst8z0d7p-1.a1.typesense.net",
            placeholder="your-cluster.a1.typesense.net",
            help="Typesense server hostname"
        )
        
        typesense_port = st.text_input(
            "Typesense Port",
            value="443",
            placeholder="443",
            help="Typesense server port (443 for HTTPS)"
        )
        
        typesense_protocol = st.selectbox(
            "Protocol",
            options=["https", "http"],
            index=0,
            help="Connection protocol"
        )
        
    
    # Reference app logic - direct file upload and processing
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        # Check required API keys
        if not all([groq_api_key, typesense_api_key]):
            st.warning("⚠️ Please provide all required API keys in the sidebar to continue.")
            return
            
        st.info(f"{len(uploaded_files)} document(s) uploaded. Processing...")
        
        # Load documents from uploaded PDFs - reference app style
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Initialize PyPDFLoader with the temporary file path
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            # Add source metadata (filename)
            for d in docs:
                d.metadata["source"] = uploaded_file.name

            documents.extend(docs)
            collection_name = sanitize_collection_name(uploaded_file.name)

        st.success(f"Loaded {len(documents)} pages from uploaded documents.")

        # Initialize Typesense client - reference app style
        try:
            typesense_client = typesense.Client({
                'nodes': [{
                    'host': typesense_host,
                    'port': int(typesense_port),
                    'protocol': typesense_protocol
                }],
                'api_key': typesense_api_key,
                'connection_timeout_seconds': 60
            })
            # Check if collection exists
            collections = typesense_client.collections.retrieve()
            collection_names = [col['name'] for col in collections]
            print(f"Collection_Name:{collection_name}")
            
            if collection_name not in collection_names:
                # Process documents with metadata extraction - reference app style
                process_documents_reference_style(documents, collection_name, typesense_client, typesense_host, typesense_port, typesense_protocol, typesense_api_key, cohere_api_key, groq_api_key, tavily_api_key)
            else:
                # Use existing collection - reference app style
                st.success(f"Using existing Typesense collection '{collection_name}'")
                setup_existing_collection_reference_style(collection_name, typesense_client, cohere_api_key, groq_api_key, tavily_api_key)
                
        except Exception as e:
            st.error(f"Error initializing Typesense client: {e}")
            st.stop()
    else:
        st.info("Please upload one or more PDF documents to start.")

if __name__ == "__main__":
    main()
