### From https://towardsdatascience.com/agentic-ai-building-autonomous-systems-from-scratch-8f80b07229ea
"""
1. Introduction
   The rapid evolution of generative AI has opened up new frontiers in automation, research, and creative tasks. 
   The global market is predicted to exceed $65 billion in 2024, with businesses increasingly adopting AI as a core strategy. 
   Key approaches like retrieval-augmented generation (RAG) and domain-specific models are becoming commonplace.

   Large Language Models (LLMs) have catalyzed this growth, not only solving complex problems but also functioning as agents. 
   Agentic AI refers to autonomous agents, typically powered by LLMs, capable of executing multi-step workflows collaboratively. 
   The demonstration provided shows how to build a multi-agent system (MAS) integrating three specialized agents:

   - A web researcher agent for ingesting and analyzing internet data.
   - A transcriptor and summarizer agent for converting video or text data into actionable summaries.
   - A blog writer agent synthesizing all gathered information into a coherent structure.

   These agents leverage foundational LLMs and existing enterprise tools, streamlining tasks, reducing human effort, and enhancing output quality.

2. AI Agents: What are they?
   AI agents are systems powered by LLMs, equipped with tools (APIs, databases, search engines) to perform tasks. 
   The input prompt might be human-written instructions or output from a previous agent. The agent’s capabilities depend on:

   -a. The Agent:
       -1. LLM choice (e.g., GPT-4, Claude Sonnet, or LLaMA variants).
       -2. The Role defining the agent’s responsibilities.
       -3. The Backstory detailing its environment and interactions.
       -4. The Goal representing what the agent must achieve as output.

   -b. The Task:
       -1. Description: Detailed instructions and constraints.
       -2. Expected Output: Defines format and content of results.

   -c. The Agent: Executes the task.
       By chaining multiple agents (a MAS), tasks can be distributed and solved collaboratively. 
       Agents can interact in sequential, hierarchical, or hybrid structures.

3. Multi-Agent Collaborative System (MAS)
   A MAS (also called a Crew) is a group of agents with complementary skills, working together toward a complex goal. 
   Structures include:

   -a. Sequential: Agents form a chain where one’s output is the next agent’s input.
   -b. Hierarchical: A leader agent delegates tasks to subordinate agents.
   -c. Hybrid: Combines sequential and hierarchical patterns.

4. CrewAI: A MAS for Writing a Blog Post
   A MAS is built to produce a blog post on “AI Agents.” Three agents work together:

5. Web Researcher Agent:
   Gathers the latest YouTube URLs and data about AI Agents using a search engine (SearXNG).
"""

## Agent and Task Definition:

researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher...

research_task:
  description: >
    Conduct thorough research about {topic}...
  expected_output: >
    A list with youtube URLs...
  agent: researcher

## Transcriptor & Summarizer Agent: Connects to YouTube API, retrieves transcripts, summarizes insights and references.
   # Agent and Task Definition:

summarizer:
  role: >
    {topic} Summarizer
  goal: >
    Summarize and extract knowledge...

summarize_task:
  description: >
    Analyse the information about {topic}...
  expected_output: >
    A json with a summary...
  agent: summarizer

## Blog Writer Agent:Uses the summary to create a detailed HTML blog post, including introduction, body, code example, and conclusion.
   # Agent and Task Definition:

blog_writer:
  role: >
    {topic} Blog Writer
  goal: >
    Create detailed blog posts...
  backstory: >
    You're a meticulous writer...

write_task:
  description: >
    Review the context...
  expected_output: >
    A fully-fledged blog post...
  agent: blog_writer

# Agents and tasks go into two YAML files: agents.yaml and tasks.yaml.

## Tools Integration
   # Search Engine (SearXNG)
   # The search tool retrieves YouTube video references:

from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
from langchain_community.utilities import SearxSearchWrapper

class SearxSearchToolInput(BaseModel):
    query: str = Field(..., description="The search query.")
    num_results: int = Field(10, description="The number of results to retrieve.")

class SearxSearchTool(BaseTool):
    name: str = "searx_search_tool"
    description: str = ("A tool to perform searches...")
    args_schema: Type[BaseModel] = SearxSearchToolInput
    _searx_wrapper: SearxSearchWrapper = PrivateAttr()

    def __init__(self, searx_host: str, unsecure: bool = False):
        super().__init__()
        self._searx_wrapper = SearxSearchWrapper(searx_host=searx_host, unsecure=unsecure)

    def _run(self, query: str, num_results: int = 10) -> List[Dict]:
        try:
            results = self._searx_wrapper.results(query=query + " :youtube", num_results=num_results)
            return results
        except Exception as e:
            return [{"Error": str(e)}]

# YouTube API Tool: Extracts transcripts and durations from YouTube videos:

from typing import Type, Optional
from pydantic import Field, BaseModel
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi
from crewai.tools import BaseTool

class YouTubeTranscriptToolInputSchema(BaseModel):
    video_url: str = Field(..., description="URL of the YouTube video.")
    language: Optional[str] = Field(None, description="Language code.")

class YouTubeTranscriptToolOutputSchema(BaseModel):
    transcript: str = Field(..., description="Transcript of the video.")
    duration: float = Field(..., description="Duration in seconds.")

class YouTubeTranscriptTool(BaseTool):
    name: str = "youtube_transcript_tool"
    description: str = ("A tool to fetch youtube transcripts...")
    args_schema: Type[BaseModel] = YouTubeTranscriptToolInputSchema

    def __init__(self):
        super().__init__()

    def _run(self, video_url: str, language: Optional[str] = None) -> YouTubeTranscriptToolOutputSchema:
        video_id = self.extract_video_id(video_url)
        try:
            if language:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            else:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            raise Exception(f"Failed to fetch transcript: {str(e)}")

        transcript_text = " ".join([t["text"] for t in transcripts])
        total_duration = sum([t["duration"] for t in transcripts])

        return YouTubeTranscriptToolOutputSchema(transcript=transcript_text, duration=total_duration)

    @staticmethod
    def extract_video_id(url: str) -> str:
        return url.split("v=")[-1].split("&")[0]

## Assembling the Crew : Using CrewAI, we define the agents, tasks, and their interaction. The crew runs the tasks in a sequential manner:

import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crew_zaai.src.crew_zaai.tools.searx import SearxSearchTool
from crew_zaai.src.crew_zaai.tools.youtube import YouTubeTranscriptTool

@CrewBase
class CrewZaai:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        search_tool = SearxSearchTool(searx_host=os.getenv("SEARXNG_BASE_URL"), unsecure=False)
        return Agent(config=self.agents_config["researcher"], tools=[search_tool], verbose=True)

    @agent
    def summarizer(self) -> Agent:
        youtube_tool = YouTubeTranscriptTool()
        return Agent(config=self.agents_config["summarizer"], tools=[youtube_tool], verbose=True)

    @agent
    def blog_writer(self) -> Agent:
        return Agent(config=self.agents_config["blog_writer"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def summarizer_task(self) -> Task:
        return Task(config=self.tasks_config["summarize_task"])

    @task
    def write_task(self) -> Task:
        return Task(config=self.tasks_config["write_task"], output_file="assets/report.html")

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True)

-----------------------------------------------------------------------------------------
import sys
import warnings
from crew_zaai.src.crew_zaai.crew import CrewZaai
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=SyntaxWarning)
load_dotenv()

inputs = {"topic": "AI Agents"}
CrewZaai().crew().kickoff(inputs=inputs)

-----------------------------------------------------------------------------------------
# makefile
"""
YOUTUBE_API_KEY=<YOUR KEY>
OPENAI_API_KEY=<YOUR KEY>
SEARXNG_BASE_URL=https://search.zaai.ai
"""
# Running the script produces an HTML blog post file in assets/, demonstrating how the three agents collaborate to research, summarize, and produce a rich blog post about AI Agents.

# Conclusion
""" Agentic AI is poised to revolutionize complex workflows by assembling multi-agent systems that can autonomously handle intricate tasks. 
    The provided demonstration shows how a MAS composed of a researcher, a summarizer, and a blog writer agent can work together end-to-end
    to produce a polished blog post, relying on foundational LLMs and simple tool integrations.

    The future holds even more possibilities, including integrating multimodal capabilities and adapting to dynamic, real-time data. 
    As investments and innovations continue, Agentic AI will redefine productivity, creativity, and problem-solving.
"""
