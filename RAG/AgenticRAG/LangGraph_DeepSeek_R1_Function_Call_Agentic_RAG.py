### From https://pub.towardsai.net/langgraph-deepseek-r1-function-call-agentic-rag-insane-results-b3f878e23a86

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import Sequence
from langchain_openai import OpenAIEmbeddings
import re
import os
import streamlit as st
import requests
from langchain.tools.retriever import create_retriever_tool

# Create Dummy Data
research_texts = [
    "Research Report: Results of a New AI Model Improving Image Recognition Accuracy to 98%",
    "Academic Paper Summary: Why Transformers Became the Mainstream Architecture in Natural Language Processing",
    "Latest Trends in Machine Learning Methods Using Quantum Computing"
]

development_texts = [
    "Project A: UI Design Completed, API Integration in Progress",
    "Project B: Testing New Feature X, Bug Fixes Needed",
    "Product Y: In the Performance Optimization Stage Before Release"
]

# Text splitting settings
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# Generate Document objects from text
research_docs = splitter.create_documents(research_texts)
development_docs = splitter.create_documents(development_texts)

# Create vector stores
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

research_vectorstore = Chroma.from_documents(
    documents=research_docs,
    embedding=embeddings,
    collection_name="research_collection"
)

development_vectorstore = Chroma.from_documents(
    documents=development_docs,
    embedding=embeddings,
    collection_name="development_collection"
)

research_retriever = research_vectorstore.as_retriever()
development_retriever = development_vectorstore.as_retriever()

research_tool = create_retriever_tool(
    research_retriever,  # Retriever object
    "research_db_tool",  # Name of the tool to create
    "Search information from the research database."  # Description of the tool
)

development_tool = create_retriever_tool(
    development_retriever,
    "development_db_tool",
    "Search information from the development database."
)

# Combine the created research and development tools into a list
tools = [research_tool, development_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage|HumanMessage|ToolMessage], add_messages]

def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]

    if isinstance(messages[0], tuple):
        user_message = messages[0][1]
    else:
        user_message = messages[0].content

    # Structure prompt for consistent text output
    prompt = f"""Given this user question: "{user_message}"
    If it's about research or academic topics, respond EXACTLY in this format:
    SEARCH_RESEARCH: <search terms>
    
    If it's about development status, respond EXACTLY in this format:
    SEARCH_DEV: <search terms>
    
    Otherwise, just answer directly.
    """

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer sk-1cddf19f9dc4466fa3ecea6fe10abec0",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data,
        verify=False
    )
    
    if response.status_code == 200:
        response_text = response.json()['choices'][0]['message']['content']
        print("Raw response:", response_text)
        
        # Format the response into expected tool format
        if "SEARCH_RESEARCH:" in response_text:
            query = response_text.split("SEARCH_RESEARCH:")[1].strip()
            # Use direct call to research retriever
            results = research_retriever.invoke(query)
            return {"messages": [AIMessage(content=f'Action: research_db_tool\n{{"query": "{query}"}}\n\nResults: {str(results)}')]}
        elif "SEARCH_DEV:" in response_text:
            query = response_text.split("SEARCH_DEV:")[1].strip()
            # Use direct call to development retriever
            results = development_retriever.invoke(query)
            return {"messages": [AIMessage(content=f'Action: development_db_tool\n{{"query": "{query}"}}\n\nResults: {str(results)}')]}
        else:
            return {"messages": [AIMessage(content=response_text)]}
    else:
        raise Exception(f"API call failed: {response.text}")

def simple_grade_documents(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    print("Evaluating message:", last_message.content)
    
    # Check if the content contains retrieved documents
    if "Results: [Document" in last_message.content:
        print("---DOCS FOUND, GO TO GENERATE---")
        return "generate"
    else:
        print("---NO DOCS FOUND, TRY REWRITE---")
        return "rewrite"

def generate(state: AgentState):
    print("---GENERATE FINAL ANSWER---")
    messages = state["messages"]
    question = messages[0].content if isinstance(messages[0], tuple) else messages[0].content
    last_message = messages[-1]

    # Extract the document content from the results
    docs = ""
    if "Results: [" in last_message.content:
        results_start = last_message.content.find("Results: [")
        docs = last_message.content[results_start:]
    print("Documents found:", docs)

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer sk-1cddf19f9dc4466fa3ecea6fe10abec0",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Based on these research documents, summarize the latest advancements in AI:
    Question: {question}
    Documents: {docs}
    Focus on extracting and synthesizing the key findings from the research papers.
    """
    
    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    print("Sending generate request to API...")
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data,
        verify=False
    )
    
    if response.status_code == 200:
        response_text = response.json()['choices'][0]['message']['content']
        print("Final Answer:", response_text)
        return {"messages": [AIMessage(content=response_text)]}
    else:
        raise Exception(f"API call failed: {response.text}")

def rewrite(state: AgentState):
    print("---REWRITE QUESTION---")
    messages = state["messages"]
    original_question = messages[0].content if len(messages)>0 else "N/A"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer sk-1cddf19f9dc4466fa3ecea6fe10abec0",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user",
            "content": f"Rewrite this question to be more specific and clearer: {original_question}"
        }],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    print("Sending rewrite request...")
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data,
        verify=False
    )
    
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    
    if response.status_code == 200:
        response_text = response.json()['choices'][0]['message']['content']
        print("Rewritten question:", response_text)
        return {"messages": [AIMessage(content=response_text)]}
    else:
        raise Exception(f"API call failed: {response.text}")

tools_pattern = re.compile(r"Action: .*")

def custom_tools_condition(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content

    print("Checking tools condition:", content)
    if tools_pattern.match(content):
        print("Moving to retrieve...")
        return "tools"
    print("Moving to END...")
    return END

workflow = StateGraph(AgentState)

# Define the workflow using StateGraph
workflow.add_node("agent", agent)
retrieve_node = ToolNode(tools)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Define nodes
workflow.add_edge(START, "agent")

# If the agent calls a tool, proceed to retrieve; otherwise, go to END

workflow.add_conditional_edges(
    "agent",
    custom_tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

# After retrieve, determine whether to generate or rewrite
workflow.add_conditional_edges("retrieve", simple_grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the workflow to make it executable
app = workflow.compile()
-----------------------------------------------------------------------------------------

def process_question(user_question, app, config):
    """Process user question through the workflow"""
    events = []
    for event in app.stream({"messages":[("user", user_question)]}, config):
        events.append(event)
    return events

def main():
    st.set_page_config(
        page_title="AI Research & Development Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        margin-top: 20px;
    }
    .data-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .research-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .dev-box {
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with Data Display
    with st.sidebar:
        st.header("📚 Available Data")
        
        st.subheader("Research Database")
        for text in research_texts:
            st.markdown(f'<div class="data-box research-box">{text}</div>', unsafe_allow_html=True)
            
        st.subheader("Development Database")
        for text in development_texts:
            st.markdown(f'<div class="data-box dev-box">{text}</div>', unsafe_allow_html=True)

    # Main Content
    st.title("🤖 AI Research & Development Assistant")
    st.markdown("---")

    # Query Input
    query = st.text_area("Enter your question:", height=100, placeholder="e.g., What is the latest advancement in AI research?")

    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("🔍 Get Answer", use_container_width=True):
            if query:
                with st.spinner('Processing your question...'):
                    # Process query through workflow
                    events = process_question(query, app, {"configurable":{"thread_id":"1"}})
                    
                    # Display results
                    for event in events:
                        if 'agent' in event:
                            with st.expander("🔄 Processing Step", expanded=True):
                                content = event['agent']['messages'][0].content
                                if "Results:" in content:
                                    # Display retrieved documents
                                    st.markdown("### 📑 Retrieved Documents:")
                                    docs_start = content.find("Results:")
                                    docs = content[docs_start:]
                                    st.info(docs)
                        elif 'generate' in event:
                            st.markdown("### ✨ Final Answer:")
                            st.success(event['generate']['messages'][0].content)
            else:
                st.warning("⚠️ Please enter a question first!")

    with col2:
        st.markdown("""
        ### 🎯 How to Use
        1. Type your question in the text box
        2. Click "Get Answer" to process
        3. View retrieved documents and final answer
        
        ### 💡 Example Questions
        - What are the latest advancements in AI research?
        - What is the status of Project A?
        - What are the current trends in machine learning?
        """)

if __name__ == "__main__":
    main()


