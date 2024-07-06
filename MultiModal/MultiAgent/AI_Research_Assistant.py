from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured_client import UnstructuredClient

#=================
# Input Files
#=================

file1 = 'unleashing-power-gen-ai-in-procurement.pdf'
file2 = 'UK_Tech_Clusters.pptx'
file3 = 'AI_In_Agriculture.docx'

#=================
# Unstructured Key
#=================

s = UnstructuredClient(
    api_key_auth="Unstructured_API_Key_Goes_Here"
)

#=================
# Partitioning Input Files
#=================

#pdf file
elements1 = partition(filename=file1)

#pptx
elements2 = partition(filename=file2)

#docx
elements3 = partition(filename=file3)


#=================
# Chunking Input Files By Title
#=================

elements = chunk_by_title(elements1 + elements2 + elements3 )

documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))


from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

from crewai import Agent, Task, Crew
from crewai_tools import BaseTool

os.environ["openai_api_key"] = "OpenAI_Key"

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

embeddings = hf
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()
handler =  StdOutCallbackHandler()

llm = ChatOpenAI(temperature=0.0,
                 model="gpt-4o",
                 max_tokens=512)

qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    callbacks=[handler],
                                    retriever=retriever)

### 
class generation_tool(BaseTool):
    name: str = "Research Paper"
    description: str = "Research Paper"

    def _run(self, argument: str) -> str:
        return qa.run("Please read, understand and analyze the provided documents")

research_tool = generation_tool()

researcher = Agent(
    role="Researcher",
    goal="Research and extract the key points from the documents",
    backstory="extract the main information and provide the main insights to the writer "
              "Don't provide any information outside the provided documents"
    ,
    verbose=True,
    tools=[research_tool],
    allow_delegation=False
    )

writer = Agent(
    role="Writer",
    goal= 'Craft easy to understand summary report',
    backstory="""You are working on writing a summary of the provided documents.
                  Use the content of the documents to develop
                  a short comprehensive summary""",
    verbose=True,
    allow_delegation=False,
)

research = Task(
    description="""Conduct an in-depth analysis of the provided documents.
                    Identify the key components and try to understand how each works.""",
    expected_output="""our final answer must be a full analysis report of the documents""",
    agent=researcher

)



write = Task(
    description=(
        "1. Use the researcher's material to craft a compelling summary"
      "2. Sections/Subtitles are properly named "
         "3. Limit the document to only 500 words "
    ),
    agent=writer,
    expected_output='a compelling summary report' 
)
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=2
)

result = crew.kickoff()



