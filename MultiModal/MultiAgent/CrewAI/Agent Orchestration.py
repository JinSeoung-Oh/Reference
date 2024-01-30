! pip install crewai

import os
from crewai import Agent, Task, Crew, Process

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-1106-preview")

Weather_reporter = Agent(
  role='Weather_reporter',
  goal="""provide historical weather 
    overall status based on the dates and location user provided.""",
  backstory="""You are a weather reporter who provides weather 
    overall status based on the dates and location user provided.
    You are using historical data from your own experience. Make your response short.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

from langchain.agents import load_tools

human_tools = load_tools(["human"])

activity_agent = Agent(
  role='activity_agent',
  goal="""responsible for activities 
    recommendation considering the weather situation from weather_reporter.""",
  backstory="""You are an activity agent who recommends 
    activities considering the weather situation from weather_reporter.
    Don't ask questions. Make your response short.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

travel_advisor = Agent(
  role='travel_advisor',
  goal="""responsible for making a travel plan by consolidating 
    the activities and require human input for approval.""",
  backstory="""After activities recommendation generated 
    by activity_agent, You generate a concise travel plan 
    by consolidating the activities.""",
  verbose=True,
  allow_delegation=False,
  tools=human_tools,
  llm=llm,
)

Insure_agent = Agent(
  role='Insure_agent',
  goal="""responsible for listing the travel plan from advisor and giving the short 
    insurance items based on the travel plan""",
  backstory="""You are an Insure agent who gives 
    the short insurance items based on the travel plan. 
    Don't ask questions. Make your response short.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

task_weather = Task(
  description="""Provide weather 
    overall status in Bohol Island in Sept.""",
  agent=Weather_reporter
)

task_activity = Task(
  description="""Make an activity list
    recommendation considering the weather situation""",
  agent=activity_agent
)
task_insure = Task(
  description="""1. Copy and list the travel plan from task_advisor. 2. giving the short 
    insurance items based on the travel plan considering its activities type and intensity.""",
  agent=Insure_agent
)

task_advisor = Task(
  description="""Make a travel plan which includes all the recommended activities, and weather,
     Make sure to check with the human if the draft is good 
     before returning your Final Answer.
    .""",
  agent=travel_advisor
)

crew = Crew(
  agents=[Weather_reporter, activity_agent,  travel_advisor, Insure_agent, ],
  tasks=[task_weather, task_activity,  task_advisor, task_insure, ],
  verbose=2
)

result = crew.kickoff()
