## From https://ai.gopubby.com/use-phidata-to-build-ai-assistants-5e0a07074b64

! pip install -U phidata
! pip install ollama duckduckgo-search openai pydantic
! pip install chardet


# Example 1 — A simple request/response assistant
from phi.assistant import Assistant
from phi.llm.ollama import Ollama

assistant = Assistant(
    llm=Ollama(model="mistral"),
    description="You are an experienced poet",
)
assistant.print_response("Write a short poem about a summer meadow", markdown=True)


####### Example 2 — searching the internet ########
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.ollama import Ollama

assistant = Assistant(
  tools=[DuckDuckGo()], 
  llm=Ollama(model="openhermes"),
  description="You are an experienced researcher",
  show_tool_calls=True)

assistant.print_response("Find the latest trending stoiries in AI ? Summarize top stories with sources.")


# Example 3 — calling an external function
from phi.llm.ollama import Ollama
from phi.tools import Toolkit
from phi.assistant import Assistant
import requests

class GetTemp(Toolkit):
    def __init__(self):
        super().__init__()

    def get_temp(self,location:str)->str:
        # Enter your API key here
        API_KEY = "YOUR_WEATHERMAP_KEY_HERE"
    
        # base_url variable to store url
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
        # Corrected the variable here to API_KEY from api_key
        complete_url = base_url + "appid=" + API_KEY + "&q=" + location
    
        # get method of requests module
        # return response object
        response = requests.get(complete_url)
        x = response.json()
     
        # Check the value of "cod" key is equal to
        # "404", means location is found otherwise,
        # city is not found
        if x["cod"] != "404":
           y = x["main"]
           # Convert temperature from Kelvin to Celsius for readability
           temp_celsius = y["temp"] - 273.15
           return str(temp_celsius)
        else:
           return None

assistant = Assistant(
    description="You are a helpful Assistant to get world temperature data using tools", 
    tools=[GetTemp().get_temp], 
    llm=Ollama(model="openhermes"),
)

assistant.print_response("What's the temperature in Edinburgh")
assistant.print_response("What's the temperature in New York")


#Example 4— Writing and running Python code
from phi.assistant.python import PythonAssistant
from phi.file.local.csv import CsvFile
from phi.tools import Toolkit
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat

import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


python_assistant = PythonAssistant(
    files=[
        CsvFile(
            path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
            description="Contains information about movies from IMDB.",
        )
    ],
    llm=OpenAIChat(model="gpt-4"),
    pip_install=True,
    
)

python_assistant.print_response("Calculate the average run-time in minutes of the movies? Your final response should be 'The average run-time of the movies is <avg_runtime> minutes' where <avg_runtime is your calculated value", markdown=True)





