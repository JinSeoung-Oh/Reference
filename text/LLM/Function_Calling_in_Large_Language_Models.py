## From https://levelup.gitconnected.com/understanding-function-calling-in-large-language-models-f182c9d62143

"""
The text discusses how function calling is revolutionizing artificial intelligence (AI) applications 
by enabling large language models (LLMs) to interact seamlessly with external tools and APIs.
This capability enhances LLMs by allowing them to perform complex tasks more effectively.

1. What is Function Calling?
   Function calling, also known as tool use, allows LLMs to connect with external tools to perform tasks that go beyond their inherent capabilities. 
   Initially introduced by OpenAI in July 2023 for their GPT models, function calling has been adopted by other AI leaders like Google and Anthropic,
   underscoring its importance in developing autonomous AI agents.

2. The Importance of Function Calling in AI
   As AI applications become more sophisticated, the ability for models to interact with external functions becomes crucial. 
   Function calling enables developers to:
   -1. Describe Functions: Define functions that an LLM can invoke based on user prompts.
   -2. Real-Time Interaction: Allow models to interact with databases, APIs, and other services in real-time.
   -3. Enhanced Responses: Provide more effective and accurate responses by accessing up-to-date information or performing calculations.
   For example, if a user asks for current weather information, the model can call a weather API to retrieve real-time data instead of relying on static knowledge.

3. Function Calling Workflow
   The workflow for function calling involves several key steps:
   -1. User Input: The process starts with a user's query or request.
   -2. Task Assessment: The LLM evaluates whether it can handle the task internally or needs to call an external function.
   -3. Function Invocation: If necessary, the model identifies and invokes the required external functions.
   -4. Output Integration: The results from the functions are integrated into the model's final response to the user.
"""

## Example code
!pip install openai duckduckgo_search --quiet

from duckduckgo_search import DDGS
import requests
from openai import OpenAI
import json
import inspect

first_tools = [
    # Tool 1 - Get Weather Information
    { 
        "type": "function",
        "function": {
            "name": "get_weather_info",
            "description": "Get the current weather information for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which to fetch weather information, i.e., city name or coordinates.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    # Tool 2 - Search Internet
    { 
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Get internet search results for real-time information",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The query to search the web for",
                    }
                },
                "required": ["search_query"],
            },
        },
    }
]

def search_internet(search_query: str) -> list:
    results = DDGS().text(str(search_query), max_results=5)
    return results

def get_weather_info(location: str) -> dict:
    api_key = "add your open weather map api key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    else:
        raise Exception(f"Failed to fetch weather information: {response.status_code}")

# Initialize the OpenAI client
client = OpenAI(api_key="add your OpenAI API key")

# Main conversation function
def run_conversation(prompt, tools, tool_choice = "auto"):
    
    messages = [{"role": "user", "content": prompt}]
    
    print("\nInitial Message: ", messages)
    
    # Send the conversation and available functions to the model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    response_message = response.choices[0].message
    print("\nResponse Message: ", response_message)
    
    tool_calls = response_message.tool_calls
    print("\nTool Calls: ", tool_calls)
    
    # Check if the model wanted to call a function
    if tool_calls:
        
        # Call the functions
        available_functions = {
            "get_weather_info": get_weather_info,
            "search_internet": search_internet,
        } 
        # extend conversation with assistant's reply
        messages.append(response_message)
        
        # Call the function and add the response
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            # Get the function signature and call the function with given arguments
            sig = inspect.signature(function_to_call)
            call_args = {
                k: function_args.get(k, v.default)
                for k, v in sig.parameters.items()
                if k in function_args or v.default is not inspect.Parameter.empty
            }
            print(f"\nCalling {function_to_call} with arguments {call_args}")
            
            function_response = str(function_to_call(**call_args))
            
            print("\nFunction Response: ", function_response)

            # Put output into a tool message
            tool_message = {
                    "tool_call_id": tool_call.id, # Needed for Parallel Tool Calling
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            print("\nAppending Message: ", tool_message)
            
            # Extend conversation with function response
            messages.append(tool_message)  

        # Get a new response from the model where it can see the entire conversation including the function call outputs
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )  

        print("\nLLM Response: ", second_response)

        print("\n---Formatted LLM Response---")
        print("\n",second_response.choices[0].message.content)
        
        return

prompt = "What's the weather like in Delhi? How about Hong Kong? Also, what's the current news in India?"

run_conversation(prompt, first_tools)

