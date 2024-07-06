# From https://medium.com/the-ai-forum/build-a-financial-analyst-agent-using-crewai-and-llamaindex-6553a035c9b8

!pip install llama-index
!pip install llama-index-llms-groq
!pip install llama-index-core
!pip install llama-index-readers-file
!pip install llama-index-tools-wolfram-alpha
!pip install llama-index-embeddings-huggingface
!pip install 'crewai[tools]'

### Setup LLM
from google.colab import userdata
from llama_index.llms.groq import Groq
from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from crewai import Agent, Task, Crew, Process

groq_api_key = userdata.get('GROQ_API_KEY')

llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)

chat_llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=1000,
)

\!wget "https://s23.q4cdn.com/407969754/files/doc_financials/2019/ar/Uber-Technologies-Inc-2019-Annual-Report.pdf" -O uber_10k.pdf

reader = SimpleDirectoryReader(input_files=["uber_10k.pdf"])
docs = reader.load_data()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(docs,
                                        embed_model=embed_model,
                                        )
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
from crewai_tools import LlamaIndexTool
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Uber 2019 10K Query Tool",
    description="Use this tool to lookup the 2019 Uber 10K Annual Report",
)
#
query_tool.args_schema.schema()

# Define your agents with roles and goals
researcher = Agent(
    role="Senior Financial Analyst",
    goal="Uncover insights about different tech companies",
    backstory="""You work at an asset management firm.
  Your goal is to understand tech stocks like Uber.""",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
    llm=chat_llm,
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
    llm=chat_llm,
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description="""Conduct a comprehensive analysis of Uber's risk factors in 2019.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the headwinds that Uber faces.
  Your post should be informative yet accessible, catering to a casual audience.
  Make it sound cool, avoid complex words.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

result = crew.kickoff()

print("######################")
print(result)



