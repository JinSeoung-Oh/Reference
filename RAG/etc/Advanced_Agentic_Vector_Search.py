## From https://towardsdev.com/implementing-advanced-agentic-vector-search-a-comprehensive-guide-to-crewai-and-qdrant-ca214ca4d039

### Below code for building server. See given link for checking folder structure  
## In the .env file
LOCAL_EMBEDDINGS="http://localhost:11434/api/embeddings"
LOCAL_EMBEDDINGS_MODEL="snowflake-arctic-embed:33m"
QDRANT_LOCAL="http://localhost:6333"
QDRANT_COLLECTION_NAME="qdrant-docs"
OPENAI_API_KEY="sk-<yourkey>"
OPENAI_LLM="gpt-4-turbo-2024-04-09"

## In the Dockerfile
FROM python:3-alpine3.19

RUN mkdir /code

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=3001"]

## /core/Tools.py
from typing import List
from crewai_tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams
from dotenv import load_dotenv, find_dotenv
import requests
import os


class QdrantTools:
    _ = load_dotenv(find_dotenv())
    # Create a Qdrant client instance
    client = QdrantClient(url=os.getenv("QDRANT_LOCAL"))

    @staticmethod
    @tool("vectorize_query")
    def vectorize_query(query: str) -> str:
        """ this is a tool to be used to vectorize the given query"""
        url = os.getenv("LOCAL_EMBEDDINGS")

        payload = {
            "model": os.getenv("LOCAL_EMBEDDINGS_MODEL"),
            "prompt": query
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)

        return response.json()

    @staticmethod
    @tool("search_qdrant")
    def search_qdrant(vector: List) -> str:
        """ this is a tool to be used to fetch the similar vectors from qdrant database"""
        # Create a search request
        result: list[ScoredPoint] = QdrantTools.client.search(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            query_vector=vector,
            search_params=SearchParams(hnsw_ef=128, exact=False),
            limit=2  # number of nearest vectors
        )
        return result


## /core/Tasks.py
from core.Agents import QdrantAgents
from crewai import Task


class QdrantTasks:
    def __init__(self):
        self.qdrant_agents = QdrantAgents()

    def vectorize_prompt_task(self, agent, prompt):
        return Task(
            description=(
                f"identify the {prompt} and return the vector embedding of the respective {prompt}"
            ),
            expected_output="A json array object with vector embeddings, example of the response as below \n"
                            "[-0.09073494374752045,-0.22729796171188354,-0.2276609241962433,0.2960631847381592].",
            agent=agent
        )

    def vector_search_task(self, agent):
        return Task(
            description=(
                "identify the top 2 similar data from the database by using the embedding from previous task."
            ),
            expected_output="A json object with similar vectors and scores associated with them",
            agent=agent
        )


## /core/Agents.py
from crewai import Agent
from langchain_openai import ChatOpenAI
from core.Tools import QdrantTools
from dotenv import load_dotenv, find_dotenv
import os


class QdrantAgents:
    def __init__(self):
        _ = load_dotenv(find_dotenv())
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # platform.openai key
        self.llm = ChatOpenAI(model=os.getenv("OPENAI_LLM"), temperature=0.3)

    def vectorizer_agent(self, prompt):
        return Agent(
            role='vectorize any given query pod names agent',
            goal=f'fetch the vector embedding for the given {prompt}',
            verbose=True,
            memory=True,
            backstory=(
                f"As a senior vector database engineer agent with qdrant vector database expertise "
                f"your job is to convert the given {prompt} into vector embedding."
            ),
            tools=[QdrantTools.vectorize_query],
            allow_delegation=True,
            llm=self.llm
        )

    def vector_search_agent(self):
        return Agent(
            role="vector search agent",
            goal="search the qdrant vector store for a given vector embedding",
            verbose=True,
            memory=True,
            backstory=("you need to search the qdrant vector database to fetch the top 2 similar data"
                       "by using the embedding from the previous agent"),
            tools=[QdrantTools.search_qdrant],
            allow_deligation=False,
            llm=self.llm
        )


## /agentic_vector_search.py
from core.Agents import QdrantAgents
from core.Tasks import QdrantTasks
from crewai import Crew, Process
from phoenix.trace.langchain import LangChainInstrumentor
import phoenix as px
import warnings

warnings.filterwarnings('ignore')

session = px.launch_app()
LangChainInstrumentor().instrument()

qdrant_agents = QdrantAgents()
qdrant_tasks = QdrantTasks()

prompt = ('That is where the accuracy matters the most. And in this case, '
          'Qdrant has proved just commendable in giving excellent search results.')

vectorizer_agent = qdrant_agents.vectorizer_agent(prompt=prompt)
vector_search_agent = qdrant_agents.vector_search_agent()

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[vectorizer_agent, vector_search_agent],
    tasks=[qdrant_tasks.vectorize_prompt_task(agent=vectorizer_agent, prompt=prompt),
           qdrant_tasks.vector_search_task(agent=vector_search_agent)],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)


# Starting the task execution process with enhanced feedback
def crew_start():
    result = crew.kickoff()
    print(px.active_session().url)
    print(result)


## ./main.py
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from agentic_vector_search import crew_start
import logging
import os

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)

if os.getenv("DEBUG_LOG_LEVEL") == "true":
    logging.basicConfig(level=logging.DEBUG)

app = FastAPI()


@app.get("/crew-kickoff")
def crew_kickoff():
    crew_start()
