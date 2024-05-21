## From https://medium.com/@suresh-kandru/llamaparse-a-deep-dive-into-financial-document-analysis-bd9d81c7ba37

import streamlit as st
import nest_asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo-0125")
Settings.llm = llm
Settings.embed_model = embed_model

documents = LlamaParse(result_type="markdown").load_data("./uber_10q_march_2022.pdf")
print(documents[0].text[:1000] + "...")

node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8)
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

recursive_index = VectorStoreIndex(nodes=base_nodes + objects)

reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large")
recursive_query_engine = recursive_index.as_query_engine(
similarity_top_k=15, node_postprocessors=[reranker], verbose=True
)

############### Streamlit Interface ###############
st.header('LlamaParse Financial Document Chat')
user_query = st.text_input("Enter your query here:", key="query1")
if st.button("Submit Query", key="submit"):
response = recursive_query_engine.query(user_query)
st.text_area("Response:", value=str(response), height=500, key="response")
###################################################

query = "How is the Cash paid for Income taxes, net of refunds from Supplemental disclosures of cash flow information?"
response = recursive_query_engine.query(query)
print(response)

query1 = "What were cash flows like from investing activities?"
response1 = recursive_query_engine.query(query1)
print(response1)




