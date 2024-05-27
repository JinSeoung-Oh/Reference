## From https://blog.stackademic.com/a-guide-to-using-semantic-cache-to-speed-up-llm-queries-with-qdrant-and-groq-dd29170c4804
## Have to add some code. It just query to semantic_cache DB, not Vecotr DB. So, If we cannot find out index in the semantic_cache DB
## it looks cannot answer the use query. 

!pip install qdrant-client langchain langchain-groq fastembed datasets

import numpy as np
import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import time
from typing import List
from fastembed import TextEmbedding
from qdrant_client.http.models import PointStruct, SearchParams
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_groq import ChatGroq

qdrant_uri = "Qdrant-URI"
qdrant_api = "Qdrant-API-Key"
groq_key = "Groq-API-key"

llm = ChatGroq(groq_api_key=groq_key,
               model='llama3-8b-8192',
               temperature=0.2) # set temperature by your own
data = load_dataset("llamafactory/PubMedQA", split="train")
data = data.to_pandas()
MAX_ROWS = 1000
OUTPUT = "output"
subset_data = data.head(MAX_ROWS)

client = QdrantClient(
    qdrant_uri,
    api_key=qdrant_api
)

chunks = subset_data[OUTPUT].to_list()
# Add data to defined collection
client.add(
   collection_name='PubMedQA',
   documents=chunks
)


class QdrantVectorStore():
    def __init__(self):
        self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = QdrantClient()
        self.db_collection_name= "PubMedQA"
       
        self.db_client = QdrantClient(
            qdrant_uri,
            api_key=qdrant_api
        )
       
        self.euclidean_threshold = threshold
   
    def get_embedding(self, question):
        embedding = list(self.encoder.embed(question))[0]
        return embedding
   
    def query_database(self,query_text):
        result = self.db_client.query(
            query_text=query_text,
            limit=3,
            collection_name=self.db_collection_name
        )
        return result
   
    def query(self, question):
        start_time = time.time()
       
        db_results = self.query_database(question)
       
        if db_results:
            response_text = db_results[0].document
            print('Retrieval without cache.')
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")
            return response_text
       
        print("No answer found in Database")
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.3f} seconds")
        return "No answer available"
      
class SemanticCache:
    def __init__(self, threshold=0.35):
        self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.cache_client = QdrantClient(":memory:")
        self.cache_collection_name = "PubMedQA-cache"


        self.cache_client.create_collection(
            collection_name=self.cache_collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance='Euclid'
            )
        )


        # Initialize Qdrant Client for external database
        self.db_client = QdrantClient(
            qdrant_uri,
            api_key=qdrant_api
        )
       
        self.db_collection_name = "PubMedQA"


        self.euclidean_threshold = threshold


    def get_embedding(self, question):
        embedding = list(self.encoder.embed(question))[0]
        return embedding


    def search_cache(self, embedding):
        search_result = self.cache_client.search(
            collection_name=self.cache_collection_name,
            query_vector=embedding,
            limit=1
        )
        return search_result


    def add_to_cache(self, question, response_text):
        # Create a unique ID for the new point
        point_id = str(uuid.uuid4())
        vector = self.get_embedding(question)
        # Create the point with payload
        point = PointStruct(id=point_id, vector=vector, payload={"response_text": response_text})
        # Upload the point to the cache
        self.cache_client.upload_points(
            collection_name=self.cache_collection_name,
            points=[point]
        )


    def query_database(self, query_text):
        results = self.db_client.query(
            query_text=query_text,
            limit=3,
            collection_name=self.db_collection_name
        )
        return results


    def ask(self, question):
        start_time = time.time()
        vector = self.get_embedding(question)
        search_result = self.search_cache(vector)
        print(search_result)
        if search_result:
            for s in search_result:
                if s.score <= self.euclidean_threshold:
                    print('Answer recovered from Cache.')
                    print(f'Found cache with score {s.score:.3f}')
                    elapsed_time = time.time() - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return s.payload['response_text']


        db_results = self.query_database(question)
        if db_results:
            response_text = db_results[0].document
            self.add_to_cache(question, response_text)
            print('Answer added to Cache.')
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")
            return response_text


        # Fallback if no response is found
        print('No answer found in Cache or Database.')
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.3f} seconds")
        return "No answer available."

## Without Semantic Cache
vector_db = QdrantVectorStore()
def chat_query(question):
    context = vector_db.query(question)
   
    prompt_template = """Answer the question with given context as per requirement.
                {context}
           
                Question: {question}"""
    prompt = PromptTemplate.from_template(prompt_template)


    chain = prompt | llm | StrOutputParser()
   
    result = chain.invoke({"question":question, "context":context})
    print("\n\n")
    print(result)

## With Semantic Cache
cache = SemanticCache()


def chat_cache(question):
    context = cache.ask(question)
   
    prompt_template = """Answer the question with given context as per requirement.
                {context}
           
                Question: {question}"""
    prompt = PromptTemplate.from_template(prompt_template)


    chain = prompt | llm | StrOutputParser()
   
    result = chain.invoke({"question":question, "context":context})
    print("\n\n")


