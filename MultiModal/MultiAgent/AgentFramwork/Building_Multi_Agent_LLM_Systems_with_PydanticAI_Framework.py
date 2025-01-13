### From https://blog.cubed.run/building-multi-agent-llm-systems-with-pydanticai-framework-a-step-by-step-guide-to-create-ai-39de7d9bb98f

## Pydantic Vs Pydantic in GenAI Vs PydanticAI
#* Pydantic
from datetime import date
from pydantic import BaseModel
class User(BaseModel):
    id: int
    name: str
    dob: date
user = User(id='123', name='Samuel Colvin', dob='1987-01-28')
#> User(id=123, name='Samuel Colvin', dob=date(1987, 1, 28))
user = User.model_validate_json('{"id: 123, "name": "Samuel Colvin", "dob": "1987-01-28"}')
#> User(id=123, name='Samuel Colvin', dob=date(1987, 1, 28))
print(User.model_json_schema())
s = {
    'properties': {
        'id': {'title': 'Id', 'type': 'integer'},
        'name': {'title': 'Name', 'type': 'string'},
        'dob': {'format': 'date', 'title': 'Dob', 'type': 'string'},
    },
    'required': ['id', 'name', 'dob'],
    'title': 'User',
    'type': 'object',
}

#* Pydantic in GenAI
from datetime import date
from pydantic import BaseModel
from openai import OpenAI
class User(BaseModel):
    """Definition of a user"""
    id: int
    name: str
    dob: date
response = OpenAI().chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': 'Extract information about the user'},
        {'role': 'user', 'content': 'The user with ID 123 is called Samuel, born on Jan 28th 87'}
    ],
    tools=[
        {
            'function': {
                'name': User.__name__,
                'description': User.__doc__,
                'parameters': User.model_json_schema(),
            },
            'type': 'function'
        }
    ]
)
user = User.model_validate_json(response.choices[0].message.tool_calls[0].function.arguments)
print(user)

#* PydanticAI
"""
PydanticAI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.
And If the user submits invalid data (e.g., "age": "twenty-five"), Pydantic will throw an error automatically
"""
from datetime import date
from pydantic_ai import Agent
from pydantic import BaseModel
class User(BaseModel):
    """Definition of a user"""
    id: int
    name: str
    dob: date
agent = Agent(
    'openai:gpt-4o',
    result_type=User,
    system_prompt='Extract information about the user',
)
result = agent.run_sync('The user with ID 123 is called Samuel, born on Jan 28th 87')
print(result.data)

## Step by Step
!pip install pydantic-ai # Requires Python 3.9+

#* Define Agent *#
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o')
result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.data)
#> Rome

async def main():
    result = await agent.run('What is the capital of France?')
    print(result.data)
    #> Paris
    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_data())
        #> London

#* Environment variable *#
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
model = GeminiModel('gemini-1.5-flash', api_key=os.environ['GEMINI_API_KEY'])
agent = Agent(model)

#* Defining Dependencies *#
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)

@agent.system_prompt  
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:  
    response = await ctx.deps.http_client.get(  
        'https://example.com',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},  
    )
    response.raise_for_status()
    return f'Prompt: {response.text}'

async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run('Tell me a joke.', deps=deps)
        print(result.data)
        #> Did you hear about the toothpaste scandal? They called it Colgate.

#* Function Tools *#
import random
from pydantic_ai import Agent, RunContext
agent = Agent(
    'gemini-1.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)

@agent.tool_plain  
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

@agent.tool  
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

dice_result = agent.run_sync('My guess is 4', deps='Anne')  
print(dice_result.data)
#> Congratulations Anne, you guessed correctly! You're a winner!

#* Results *#
from pydantic import BaseModel
from pydantic_ai import Agent
class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent('gemini-1.5-flash', result_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""

#* Using Messages as Input for Further Agent Runs *#
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.data)
#> This is an excellent joke invent by Samuel Colvin, it needs no explanation.
print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.', part_kind='system-prompt'
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.',
                part_kind='text',
            )
        ],
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invent by Samuel Colvin, it needs no explanation.',
                part_kind='text',
            )
        ],
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""

#* write unit tests *#
import asyncio
from datetime import date
from pydantic_ai import Agent, RunContext
from fake_database import DatabaseConn  
from weather_service import WeatherService  
weather_agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():  
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)

async def run_weather_forecast(  
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:
        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.data)
        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )

#* TestModel *#
from datetime import timezone
import pytest
from dirty_equals import IsNow
from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ArgsDict,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    ModelRequest,
)
from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent
pytestmark = pytest.mark.anyio  
models.ALLOW_MODEL_REQUESTS = False  

async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):  
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)  
    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'  
    assert messages == [  
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Providing a weather forecast at the locations the user provides.',
                ),
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),  
                ),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args=ArgsDict(
                        args_dict={
                            'location': 'a',
                            'forecast_date': '2024-01-01',  
                        }
                    ),
                    tool_call_id=None,
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=None,
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
        ),
    ]

#* FunctionModel *#
import re
import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False

def call_weather_forecast(  
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}  
        return ModelResponse(
            parts=[ToolCallPart.from_raw_args('weather_forecast', args)]
        )
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse.from_text(f'The forecast is: {msg.content}')

async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):  
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)
    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'


########### A Simple AI Agent
from pydantic_ai import Agent
from pydantic import BaseModel
# Define the structure of the response
class CityInfo(BaseModel):
 city: str
 country: str
# Create an agent
agent = Agent(
 model='openai:gpt-4o', # Specify your model
 result_type=CityInfo # Enforce the structure of the response
)
# Run the agent
if __name__ == '__main__':
 result = agent.run_sync("Tell me about Paris.")
 print(result.data) # Outputs: {'city': 'Paris', 'country': 'France'}


############ Adding Tools to Your Agent
from pydantic_ai import Agent, RunContext
import random
# Define the agent
agent = Agent('openai:gpt-4o')
# Add a tool to roll a die
@agent.tool
async def roll_die(ctx: RunContext, sides: int = 6) -> int:
    """Rolls a die with the specified number of sides."""
    return random.randint(1, sides)
# Run the agent
if __name__ == '__main__':
    result = agent.run_sync("Roll a 20-sided die.")
    print(result.data)  # Outputs a random number between 1 and 20

########### Example
GEMINI_API_KEY = "Your api key"
TVLY_API_KEY ="Your api key"

!pip install 'pydantic-ai[examples]' \
pip install tavily-python

import asyncio
from dataclasses import dataclass
import json
import os
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext 
from dotenv import load_dotenv
from pydantic_ai.models.gemini import GeminiModel
from httpx import AsyncClient
from tavily import TavilyClient
load_dotenv()
@dataclass
class Deps:
    content_strategist_agent:Agent[None,str]
    client: AsyncClient
    tvly_api_key: str | None
    content:str
    
@dataclass
class Content:
    points: str
class BlogPostBaseModel(BaseModel):
    content:str
model = GeminiModel('gemini-1.5-flash', api_key=os.environ['GEMINI_API_KEY'])
# Agent setup
search_agent = Agent(
    model= model ,
    result_type=Content,
    system_prompt=(
        """you are Senior Research Analyst and your work as a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.given topic pydantic AI.
  Full analysis report in bullet points"""
    ),
    retries=2
)
content_writer_agents = Agent(
    model= model ,
    deps_type=Content,
    result_type=BlogPostBaseModel,
    system_prompt=(
        """You are a renowned Content Strategist, known for your insightful and engaging articles.use search_web for getting the list of points
  You transform complex concepts into compelling narratives.Full blog post of at least 4 paragraphs include paragrahs,headings, bullet points include html tags, please remove '\\n\\n'}"""
    ),
    retries=2
)
# Web Search for your query
@search_agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """Web Search for your query."""
    tavily_client = TavilyClient(api_key=ctx.deps.tvly_api_key)
    response =  tavily_client.search(web_query)
    return json.dumps(response)
@search_agent.tool
async def content_writer_agent(
    ctx: RunContext[Deps], question: str
) -> str:
    """Use this tool to communicate with content strategist"""
    print(question)
    response = await ctx.deps.content_strategist_agent.run(user_prompt=question)
    ctx.deps.content = response.data
    print("contentstragist") 
    return response.data
async def main():
    async with AsyncClient() as client:
        message_history =[]
        tvly_api_key = os.environ['TVLY_API_KEY']
        deps = Deps(client=client, tvly_api_key=tvly_api_key,content_strategist_agent=content_writer_agents,content="")
        result = await search_agent.run("blog article for Pydantic AI",message_history=message_history, deps=deps)
        message_history = result.all_messages()
        print("Blog:")
        print(deps.content)
if __name__ == '__main__':
    asyncio.run(main()



