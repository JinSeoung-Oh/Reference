### From https://medium.com/data-science-collective/langchain-mcp-rag-ollama-the-key-to-powerful-agentic-ai-91529b2fa320

pip install -r requirements.txt

#mcp_server.py
import asyncio
from mcp.server.fastmcp import FastMCP
import rag
import search
import logging
import os

mcp = FastMCP(
    name="web_search", 
    version="1.0.0",
    description="Web search capability using Exa API , Firecrawl API  that provides real-time internet search results and use RAG to search for relevant data. Supports both basic and advanced search with filtering options including domain restrictions, text inclusion requirements, and date filtering. Returns formatted results with titles, URLs, publication dates, and content summaries."
)

@mcp.tool()
async def search_web_tool(query: str) -> str:
    logger.info(f"Searching web for query: {query}")
    formatted_results, raw_results = await search.search_web(query)
    
    if not raw_results:
        return "No search results found."
    
    urls = [result.url for result in raw_results if hasattr(result, 'url')]
    if not urls:
        return "No valid URLs found in search results."
        
    vectorstore = await rag.create_rag(urls)
    rag_results = await rag.search_rag(query, vectorstore)
    
    # You can optionally include the formatted search results in the output
    full_results = f"{formatted_results}\n\n### RAG Results:\n\n"
    full_results += '\n---\n'.join(doc.page_content for doc in rag_results)
    
    return full_results

@mcp.tool()
async def get_web_content_tool(url: str) -> str:
    try:
        documents = await asyncio.wait_for(search.get_web_content(url), timeout=15.0)
        if documents:
            return '\n\n'.join([doc.page_content for doc in documents])
        return "Unable to retrieve web content."
    except asyncio.TimeoutError:
        return "Timeout occurred while fetching web content. Please try again later."
    except Exception as e:
        return f"An error occurred while fetching web content: {str(e)}"

----------------------------------------------------------------------------------------------------------

#rag.py

from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import search
from langchain_core.documents import Document
import os
import asyncio

async def create_rag(links: list[str]) -> FAISS:
    try:
        model_name = os.getenv("MODEL", "text-embedding-ada-002") 
        # Change any embedding you want, whether Ollama or MistralAIEmbeddings
        # embeddings = MistralAIEmbeddings(
        #     model="mistral-embed",
        #     chunk_size=64
        # )
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            chunk_size=64
        )
        documents = []
        # Use asyncio.gather to process all URL requests in parallel
        tasks = [search.get_web_content(url) for url in links]
        results = await asyncio.gather(*tasks)
        for result in results:
            documents.extend(result)
        
        # Text chunking processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)
        # print(documents)
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag: {str(e)}")
        raise

async def create_rag_from_documents(documents: list[Document]) -> FAISS:
    """
    Create a RAG system directly from a list of documents to avoid repeated web scraping
    
    Args:
        documents: List of already fetched documents
        
    Returns:
        FAISS: Vector store object
    """
    try:
        model_name = os.getenv("MODEL")
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            chunk_size=64
        )
        
        # Text chunking processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag_from_documents: {str(e)}")
        raise

---------------------------------------------------------------------------------------------

#search.py

import asyncio
from dotenv import load_dotenv
import os
from exa_py import Exa
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
import requests

# Load .env variables
load_dotenv(override=True)

# Initialize the Exa client
exa_api_key = os.getenv("EXA_API_KEY", " ")
exa = Exa(api_key=exa_api_key)
os.environ['FIRECRAWL_API_KEY'] = ''
# Default search config
websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

# Constants for web content fetching
MAX_RETRIES = 3
FIRECRAWL_TIMEOUT = 30  # seconds

async def search_web(query: str, num_results: int = None) -> Tuple[str, list]:
    """Search the web using Exa API and return both formatted results and raw results."""
    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }

        search_results = exa.search_and_contents(
            query,
            summary={"query": "Main points and key takeaways"},
            **search_args
        )

        formatted_results = format_search_results(search_results)
        return formatted_results, search_results.results
    except Exception as e:
        return f"An error occurred while searching with Exa: {e}", []

def format_search_results(search_results):
    if not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"
    for idx, result in enumerate(search_results.results, 1):
        title = result.title if hasattr(result, 'title') and result.title else "No title"
        url = result.url
        published_date = f" (Published: {result.published_date})" if hasattr(result, 'published_date') and result.published_date else ""

        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"

        if hasattr(result, 'summary') and result.summary:
            markdown_results += f"> **Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"

    return markdown_results

async def get_web_content(url: str) -> List[Document]:
    """Get web content and convert to document list."""
    for attempt in range(MAX_RETRIES):
        try:
            # Create FireCrawlLoader instance
            loader = FireCrawlLoader(
                url=url,
                mode="scrape"
            )
            
            # Use timeout protection
            documents = await asyncio.wait_for(loader.aload(), timeout=FIRECRAWL_TIMEOUT)
            
            # Return results if documents retrieved successfully
            if documents and len(documents) > 0:
                return documents
            
            # Retry if no documents but no exception
            print(f"No documents retrieved from {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)  # Wait 1 second before retrying
                continue
                
        except requests.exceptions.HTTPError as e:
            if "Website Not Supported" in str(e):
                # Create a minimal document with error info
                print(f"Website not supported by FireCrawl: {url}")
                content = f"Content from {url} could not be retrieved: Website not supported by FireCrawl API."
                return [Document(page_content=content, metadata={"source": url, "error": "Website not supported"})]
            else:
                print(f"HTTP error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})")
                
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            raise
        except Exception as e:
            print(f"Error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            raise
    
    # Return empty list if all retries failed
    return []
  -----------------------------------------------------------------------------------------------------
#agent.py

import asyncio
import os
import sys

# Import search and RAG modules directly
import search
import rag

async def main():
    
    # Get query from command line or input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter search query: ")
    
    print(f"Searching for: {query}")
    
    try:
        # Call search directly
        formatted_results, raw_results = await search.search_web(query)
        
        if not raw_results:
            print("No search results found.")
            return
        
        print(f"Found {len(raw_results)} search results")
        
        # Extract URLs
        urls = [result.url for result in raw_results if hasattr(result, 'url')]
        if not urls:
            print("No valid URLs found in search results.")
            return
            
        print(f"Processing {len(urls)} URLs")
        
        # Create RAG
        vectorstore = await rag.create_rag(urls)
        rag_results = await rag.search_rag(query, vectorstore)
        
        # Format results
        print("\n=== Search Results ===")
        print(formatted_results)
        
        print("\n=== RAG Results ===")
        for doc in rag_results:
            print(f"\n---\n{doc.page_content}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())


