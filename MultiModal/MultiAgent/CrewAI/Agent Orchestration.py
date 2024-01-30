! pip install crewai

import os
from crewai import Agent, Task, Crew, Process

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-1106-preview")

