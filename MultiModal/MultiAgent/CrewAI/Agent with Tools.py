! pip install duckduckgo_search

## With Pre-built tools
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
activity_agent = Agent(
  role='activity_agent',
  goal="""responsible for actitivies 
    recommendation considering the weather situation from weather_reporter.""",
  backstory="""You are an activity agent who recommends 
    activities considering the weather situation from weather_reporter.
    Don't ask questions. Make your response short.""",
  verbose=True,
  allow_delegation=False,
  
  tools=[search_tool],

  llm=llm,
)

task2 = Task(
  description="""Make a research for suitable and up-to-date activities 
    recommendation considering the weather situation""",
  agent=activity_agent
)

## Custom tools
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field

class WeatherInput(BaseModel):
    search_string: str = Field(description="the search string for the weather status")

def get_weather(search_string:str) -> str:
    """Look up the weather status"""
    return "It's raining season with typhoons."

weather_search = StructuredTool.from_function(
    func=get_weather,
    name="weather_search",
    description="search for the weather status",
    args_schema=WeatherInput,
    return_direct=True,
)

Weather_reporter = Agent(
  role='Weather_reporter',
  goal="""providing weather 
    overall status based on the dates and location the user provided.""",
  backstory="""You are a weather reporter who provides weather 
    overall status based on the dates and location the user provided.
    Make your response short.""",
  verbose=True,
  allow_delegation=False,
  tools=[weather_search],
  llm=llm,
)

task1 = Task(
  description="""providing weather 
    overall status in Bohol Island in September.""",
  agent=Weather_reporter
)

