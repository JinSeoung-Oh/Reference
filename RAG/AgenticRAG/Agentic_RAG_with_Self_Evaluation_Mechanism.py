### From https://medium.com/data-science-collective/how-to-create-agentic-rag-with-self-evaluation-mechanism-73097f73d262

import fitz  
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
import os

os.environ["TAVILY_API_KEY"] = "your-api-key"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

text = extract_text_from_pdf("pdf_files/article_2.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([text])

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

### Agentic RAG
llm = ChatOpenAI(model="gpt-4")
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

search = TavilySearchResults(max_results=2)

def query_reformulation(query):
    response = llm.predict("Rewrite this query to be more specific: " + query)
    return response

def self_evaluate(input_text):
    parts = input_text.split("|||")
    query = parts[0]
    response = parts[1]
    sources = parts[2] if len(parts) > 2 else ""
    
    evaluation_prompt = f"""
    Evaluate the following response to the query:
    
    QUERY: {query}
    RESPONSE: {response}
    SOURCES: {sources}
    
    Assess based on:
    1. Factual accuracy (Does it match the sources?)
    2. Completeness (Does it address all aspects of the query?)
    3. Relevance (Is the information relevant to the query?)
    4. Hallucination (Does it contain information not supported by sources?)
    
    Return a confidence score from 0-10 and explanation.
    """
    
    evaluation = llm.predict(evaluation_prompt)
    return evaluation

tools = [
    Tool(
        name="Article Retrieval",
        func=lambda q: retrieval_qa_chain({"query": q})["result"],
        description="Retrieve knowledge from the article database."
    ),
    
    Tool(
        name="Web search",
        func=search,
        description="If the requested information cannot be found in the documents, it specifies this and performs a web search."
    ),
    Tool(
        name="Query reformulation",
        func=query_reformulation,
        description="Reformulate a query to be more specific and targeted."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def get_evaluated_response(query):
    response = agent.run(query)
    
    try:
        result = retrieval_qa_chain({"query": query})
        sources = [doc.page_content for doc in result.get("source_documents", [])]
        sources_text = "\n".join(sources)
    except Exception as e:
        sources_text = "No sources available"
    
    evaluation = self_evaluate(f"{query}|||{response}|||{sources_text}")
    
    return {
        "query": query,
        "response": response,
        "evaluation": evaluation,
        "sources": sources_text
    }

def transparent_response(query):
    result = get_evaluated_response(query)
    
    return f"""
    Response: {result['response']}
    
    Confidence assessment: {result['evaluation']}
    """

print(transparent_response("What is multi agent system?"))


