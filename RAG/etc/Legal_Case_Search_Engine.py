## From https://ai.gopubby.com/building-a-legal-case-search-engine-using-qdrant-llama-3-langchain-and-exploring-different-655ed5b25f30
## Just see the Filtering Techniques

!pip install langchain-huggingface
!pip install qdrant-client
!pip install langchain-qdrant
!pip install langchain-community
!pip install lark

import json
dataset = json.load(open("dataset.json"))['cases_list']
###################################
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
                            model_name=model_name,
                            model_kwargs=model_kwargs,
                            encode_kwargs=encode_kwargs
                           )
hf.embed_query("demonetization case")


#### LangChain Self-Query Retriever
langchain_dataset = json.load(open("dataset.json"))['cases_list']
import re
from langchain_core.documents import Document
from langchain_community.vectorstores.qdrant import Qdrant

def remove_punctuation(text):
    no_punct_text = re.sub(r'[^\w\s,]', '', text)
    return no_punct_text.lower()

for x in langchain_dataset:
    for key, val in x['metadata'].items():
        x['metadata'][key] = remove_punctuation(str(val))
    for val in x['page_content']:
        x['page_content'] = remove_punctuation(str(x['page_content']))

for i, x in enumerate(langchain_dataset):
    langchain_dataset[i] = Document(
        page_content=x['page_content'],
        metadata=x['metadata'],
    )


vectorstore = Qdrant.from_documents(langchain_dataset, hf, 
                                    location=":memory:",  
                                    collection_name="langchain_legal",)


############## Langchain self query filter ##############
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_groq import ChatGroq

metadata_field_info = [
    AttributeInfo(
        name="court",
        description="The judiciary court's name.",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the court decision was given.",
        type="string",
    ),
    AttributeInfo(
        name="judges",
        description="The list of names of the case judges.",
        type="string",
    ),
    AttributeInfo(
        name="legal_topics",
        description="list of the topic names of the case",
        type="string",
    ),
    AttributeInfo(
        name="relevant_laws",
        description="list of relevant laws that applied on the case decision",
        type="string",
    )
]
document_content_description = "Brief summary of a court case"
llm = ChatGroq(temperature=0,api_key="your-groq-key")
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)


############## Qdrant Payload filter ##############
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client import models


client = QdrantClient(":memory:")


client.recreate_collection(
    collection_name="law_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# NOTE: consider splitting the data into chunks to avoid hitting the server's payload size limit
# or use `upload_collection` or `upload_points` methods which handle this for you
# WARNING: uploading points one-by-one is not recommended due to requests overhead
client.upsert(
    collection_name="law_docs",
    points=[
        PointStruct(
            id=idx,
            vector=hf.embed_query(element['page_content']),
            payload=element['metadata']
        )
        for idx, element in enumerate(dataset)
    ]
)

client.scroll(
    collection_name="law_docs",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="year",
                match=models.MatchValue(value=2023),
            ),
            # models.FieldCondition(
            #     key="color",
            #     match=models.MatchValue(value="red"),
            # ),
        ]
    ),
)

############# #Semantic filtering ##############
metadata_fields = [x['metadata'] for x in dataset]
metadata_list = []
for elem in metadata_fields:
    s = ''
    for key in elem:
        s = s + f"{key} : {elem[key]}\n"
    s = s.strip().lower().replace('.','').replace("'",'').replace('[','').replace(']','')
    # s = remove_punctuation(s)
    metadata_list.append(s)

metadata_client = QdrantClient(":memory:")
metadata_client.recreate_collection(
    collection_name="metadata",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

metadata_client.upsert(
    collection_name="metadata",
    points=[
        PointStruct(
            id=idx,
            vector=hf.embed_query(element),
        )
        for idx, element in enumerate(metadata_list)
    ]
)

def find_hit_details(hit_list, hit):
    for i, x in enumerate(hit_list):
        if hit == x.id:
            return i
    return -1

def semantic_filtering(text):
    first_level = set()
    second_level = set()
    matching_hits = {}

    query_vector = hf.embed_query(text)
    hits = client.search("law_docs", query_vector, limit=5)
    for h in hits:
        first_level.add(h.id)
    
    filter_hits = metadata_client.search("metadata", query_vector, limit=5)
    filter_hits_dict = {fh.id: fh for fh in filter_hits}
    for fh in filter_hits:
        second_level.add(fh.id)
    
    common_hits = first_level & second_level
    for hit in common_hits:
        filter_hit_detail = filter_hits_dict[hit]
        if filter_hit_detail.score > 0.65:
            matching_hits[filter_hit_detail.score] = hit

    sorted_matching_hits = sorted(matching_hits.items(), reverse=True)
    
    if sorted_matching_hits:
        print("semantic_filtering")
        return [dataset[hit] for score, hit in sorted_matching_hits]
    else:
        print("No filter found")
        return [dataset[hit] for hit in first_level]


