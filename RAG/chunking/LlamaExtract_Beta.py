## From https://pub.towardsai.net/introducing-llamaextract-beta-transforming-metadata-extraction-for-enhanced-rag-queries-de3d74d34cd7

! pip install llama-extract

import nest_asyncio
import os
from llama_extract import LlamaExtract
from pydantic import BaseModel, Field

nest_asyncio.apply()
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."

SCHEMA_NAME = "payslip_SCHEMA"
extractor = LlamaExtract()

class PayslipMetadata(BaseModel):
    """paylip metadata."""

    Date_of_Joining : str = Field(
        ..., description="this is employee date of join"
    )
    employee_name: str = Field(
        ...,
        description="this is payslip employee name",
    )
    department: str = Field(
        ..., description="this is payslip department"
    )
PayslipMetadata.schema()

extraction_schema = extractor.create_schema("Test Schema", PayslipMetadata)
# specify your own pdf path
extractions = extractor.extract(extraction_schema.id, ["./file3.pdf"])


### Infer metadata using LlamaExtract
extraction_schema = await extractor.ainfer_schema(
    "Test Schema",["./test_extraction.pdf"]
)
extraction_schema.data_schema

extractions = await extractor.aextract(
    extraction_schema.id,
    [ "./test_extraction.pdf"],
)
extractions[0].data


### Load the documents for RAG
! pip uninstall llama-index 
! pip install -U llama-index --upgrade --no-cache-dir --force-reinstall --user
! pip install llama-parse

from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
)

def model_answer(key: str, value: str, query: str, similarity_top_k: int = 5):
 
        
    filters = MetadataFilters(
    filters=[
        MetadataFilter(key=key, value=value),

    ],
        condition="and",
    )


    query_engine = index.as_query_engine(
                                 similarity_top_k=similarity_top_k,
                                     filters = filters,
                                   )

    response = query_engine.query(query)
    return response.response
  
parser = LlamaParse(
    result_type="text",
    verbose=True,
    language="en",
    num_workers=2,
)

for index in range(len(pdf_documents)):
    
    pdf_documents[index].metadata["payslip_employee_name"]=extractions[0].data["employee_name"]
    pdf_documents[index].metadata["payslip_department"]=extractions[0].data["department"]
    pdf_documents[index].metadata["payslip_Date_of_Joining"]=extractions[0].data["Date_of_Joining"]

    pdf_documents[index].text_template = "Metadata: {metadata_str}\n-----\nContent: {content}"
    pdf_documents[index].metadata_seperator="::"
    pdf_documents[index].metadata_template="{key}=>{value}"

api_key = "your openai api"
os.environ["OPENAI_API_KEY"] = api_key
llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

index = VectorStoreIndex(pdf_documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)
# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever)
response = query_engine.query("hello")

###
query="Help me to summarize the payslip for employee name Sally Harley  "
model_answer(key="payslip_department", value='Marketing', query=query, similarity_top_k=5)


