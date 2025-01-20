### From https://generativeai.pub/agentic-rag-series-exploring-langgraph-for-advanced-workflows-60152c017795

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
import gradio as gr
import tempfile
import validators
from io import StringIO

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def process_sources(urls=None, pdf_files=None):
    """Process both URLs and PDF files"""
    docs_list = []
    
    # Handle URLs
    if urls and urls.strip():
        url_list = [url.strip() for url in urls.split(",")]
        for url in url_list:
            if validators.url(url):
                try:
                    url_docs = WebBaseLoader(url).load()
                    docs_list.extend(url_docs)
                except Exception as e:
                    print(f"Error loading URL {url}: {e}")
    
    # Handle PDFs
    if pdf_files:
        for pdf in pdf_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(pdf.read())
                    loader = PyPDFLoader(tmp.name)
                    docs_list.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF: {e}")

def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")
    
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    model = ChatOllama(temperature=0, model="llama3.2", streaming=True)
    llm_with_tool = model.with_structured_output(grade)
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Document: {context}
        Question: {question}
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    chain = prompt | llm_with_tool
    
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    
    scored_result = chain.invoke({"question": question, "context": docs})
    return "generate" if scored_result.binary_score == "yes" else "rewrite"

def agent(state):
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOllama(temperature=0, streaming=True, model="llama3.2")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    msg = [HumanMessage(content=f"""
    Look at the input and try to reason about the underlying semantic intent / meaning.
    Initial question: {question}
    Formulate an improved question:""")]
    
    model = ChatOllama(temperature=0, model="llama3.2", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(temperature=0, streaming=True, model="llama3.2")
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def process_query(urls, pdf_files, query):
    docs_list = process_sources(urls, pdf_files)
    if not docs_list:
        return "No valid documents provided. Please input URLs or upload PDFs."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs_list)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )
    retriever = vectorstore.as_retriever()
    
    global tools
    tools = [create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and return information from the provided documents."
    )]
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    graph = workflow.compile()
    
    inputs = {"messages": [HumanMessage(content=query)]}
    response = ""
    for output in graph.stream(inputs):
        for key, value in output.items():
            if value.get("messages"):
                response = value["messages"][-1].content
    
    return response

def create_interface():
    with gr.Blocks(title="Document Q&A Assistant") as interface:
        gr.Markdown("# Document Q&A Assistant")
        gr.Markdown("*You can provide URLs, PDF files, or both*")
        
        with gr.Tab("Upload Documents"):
            urls = gr.Textbox(label="Enter URLs (comma separated)", placeholder="https://example1.com, https://example2.com")
            pdfs = gr.File(file_count="multiple", label="Upload PDF files", file_types=[".pdf"])
            upload_btn = gr.Button("Process Documents")
            upload_status = gr.Textbox(label="Upload Status")
            
            def handle_upload(urls, pdfs):
                if not urls and not pdfs:
                    return "Please provide either URLs, PDF files, or both"
                docs = process_sources(urls, pdfs)
                if docs:
                    return "Documents processed successfully!"
                return "No valid documents provided. Please input valid URLs or PDF files"
            
            upload_btn.click(
                fn=handle_upload,
                inputs=[urls, pdfs],
                outputs=upload_status
            )
        
        with gr.Tab("Chat"):
            query = gr.Textbox(label="Ask a question about the documents")
            chat_btn = gr.Button("Ask")
            response = gr.Textbox(label="Response")
            
            chat_btn.click(
                fn=process_query,
                inputs=[urls, pdfs, query],
                outputs=response
            )
            
    return interface

interface = create_interface()

if __name__ == "__main__":
    interface.launch()


