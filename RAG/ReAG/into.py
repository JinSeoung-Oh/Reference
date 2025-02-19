### From https://medium.com/@nayakpplaban/fixing-rag-with-reasoning-augmented-generation-919939045789

"""
1. Overview
   -a. The Problem with Traditional RAG:
       -1. Shallow Retrieval: Traditional Retrieval-Augmented Generation (RAG) systems act like “librarians 
           with bad memories”—they rely on semantic search to retrieve documents based on surface-level similarities,
           often missing contextually relevant information.
       -2. Complex Infrastructure: The RAG pipeline involves chunking, embedding, and maintaining vector databases,
           which can lead to errors (stale indexes, mismatched splits) and add operational complexity.
       -3. Static Knowledge: Updating indexed documents is slow, which is problematic in dynamic fields
           (e.g., medicine, finance).

2. Introducing ReAG (Reasoning-Augmented Generation)
   -a. Core Idea:
       -1. ReAG rethinks the retrieval process by having the language model reason over raw documents
           (text files, PDFs, URLs) without the traditional preprocessing into embeddings or chunks.
   -b. How It Works:
       -1. Raw Ingestion: Documents are fed as-is into the model, preserving the full context.
       -2. Dual Questions: The model asks:
           -1) “Is this document useful?” (relevance check)
           -2) “What specific parts matter?” (content extraction)
   -c. Synthesis: It then synthesizes the extracted insights—much like a human researcher connecting dots
                  between different pieces of information.

3. Key Advantages and Trade-offs of ReAG
   -a. Strengths:
       -1. Dynamic Data Handling: Processes real-time updates (e.g., news, market data) without needing constant
                                  re-embedding.
       -2. Complex Query Resolution: Better at answering nuanced questions that require inference across multiple
                                     documents.
       -3. Multimodal Capabilities: Can analyze and synthesize from various formats (text, charts, tables) 
                                    without extra preprocessing.
   -b. Trade-offs:
       -1. Cost: Requires multiple LLM calls (one per document), which is more expensive than vector searches.
       -2. Scalability: May be slower for very large datasets (e.g., millions of documents); hybrid solutions 
                        (combining RAG for filtering with ReAG for deep analysis) might be necessary.

4. The Future of ReAG
   -a. Hybrid Systems: Combining RAG for initial filtering with ReAG for deep reasoning to balance cost
                       and thoroughness.
   -b. Cheaper Models: Open-source and quantized LLMs will further reduce costs.
   -c. Bigger Context Windows: Future models with even larger context windows will enhance ReAG’s ability 
                               to process extensive documents (potentially billions of tokens).

5. Final Takeaway
   ReAG isn’t about completely replacing RAG—it’s about reimagining how AI interacts with knowledge.
   By treating retrieval as a reasoning task, ReAG enables a more holistic, nuanced approach akin to human research. 
   It addresses the inherent limitations of traditional RAG systems, offering dynamic, context-driven insights
   that can power smarter, more robust AI applications.
"""

!pip install langchain langchain_groq langchain_ollama langchain_community pymupdf pypdf

!mkdir ./data
!mkdir ./chunk_caches
!wget "https://www.binasss.sa.cr/int23/8.pdf" -O "./data/fibromyalgia.pdf"

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os
from pydantic import BaseModel,Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

os.environ["GROQ_API_KEY"] = "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"

llm_relevancy = ChatGroq(
     model="llama-3.3-70b-versatile",
    temperature=0,)

llm = ChatOllama(model="deepseek-r1:14b",
                 temperature=0.6,
                 max_tokens=3000,
                )

REAG_SYSTEM_PROMPT = """
# Role and Objective
You are an intelligent knowledge retrieval assistant. Your task is to analyze provided documents or URLs to extract the most relevant information for user queries.

# Instructions
1. Analyze the user's query carefully to identify key concepts and requirements.
2. Search through the provided sources for relevant information and output the relevant parts in the 'content' field.
3. If you cannot find the necessary information in the documents, return 'isIrrelevant: true', otherwise return 'isIrrelevant: false'.

# Constraints
- Do not make assumptions beyond available data
- Clearly indicate if relevant information is not found
- Maintain objectivity in source selection
"""

rag_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

class ResponseSchema(BaseModel):
    content: str = Field(...,description="The page content of the document that is relevant or sufficient to answer the question asked")
    reasoning: str = Field(...,description="The reasoning for selecting The page content with respect to the question asked")
    is_irrelevant: bool = Field(...,description="Specify 'True' if the content in the document is not sufficient or relevant to answer the question asked otherwise specify 'False' if the context or page content is relevant to answer the question asked")


class RelevancySchemaMessage(BaseModel):
    source: ResponseSchema

def format_doc(doc: Document) -> str:
    return f"Document_Title: {doc.metadata['title']}\nPage: {doc.metadata['page']}\nContent: {doc.page_content}"

def extract_relevant_context(question,documents):
    result = []
    for doc in documents:
        formatted_documents = format_doc(doc)
        system = f"{REAG_SYSTEM_PROMPT}\n\n# Available source\n\n{formatted_documents}"
        prompt = f"""Determine if the 'Avaiable source' content supplied is sufficient and relevant to ANSWER the QUESTION asked.
        QUESTION: {question}
        #INSTRUCTIONS TO FOLLOW
        1. Analyze the context provided thoroughly to check its relevancy to help formulizing a response for the QUESTION asked.
        2, STRICTLY PROVIDE THE RESPONSE IN A JSON STRUCTURE AS DESCRIBED BELOW:
            ```json
               {{"content":<<The page content of the document that is relevant or sufficient to answer the question asked>>,
                 "reasoning":<<The reasoning for selecting The page content with respect to the question asked>>,
                 "is_irrelevant":<<Specify 'True' if the content in the document is not sufficient or relevant.Specify 'False' if the page content is sufficient to answer the QUESTION>>
                 }}
            ```
         """
        messages =[ {"role": "system", "content": system},
                       {"role": "user", "content": prompt},
                    ]
        response = llm_relevancy.invoke(messages)    
        print(response.content)
        formatted_response = relevancy_parser.parse(response.content)
        result.append(formatted_response)
    final_context = []
    for items in result:
        if (items['is_irrelevant'] == False) or ( items['is_irrelevant'] == 'false') or (items['is_irrelevant'] == 'False'):
            final_context.append(items['content'])
    return final_context

def generate_response(question,final_context):
    prompt = PromptTemplate(template=rag_prompt,
                                     input_variables=["question","context"],)
    chain  = prompt | llm
    response = chain.invoke({"question":question,"context":final_context})
    print(response.content.split("\n\n")[-1])
    return response.content.split("\n\n")[-1]
  
relevancy_parser = JsonOutputParser(pydantic_object=RelevancySchemaMessage)

file_path = "./data/fibromyalgia.pdf"
loader = PyMuPDFLoader(file_path)
#
docs = loader.load()

question = "What is Fibromyalgia?"
final_context = extract_relevant_context(question,docs)

final_response = generate_response(question,final_context)
final_response
