## From https://levelup.gitconnected.com/llama-3-2-vision-model-tutorial-build-vision-apps-multimodal-agents-17ff91b36bca

!pip install openai instructor pydantic

from openai import OpenAI
from os import getenv
import instructor
from pydantic import BaseModel, Field

class MovieRate(BaseModel):
    movie_name: str = Field(..., description="The name of the movie on the poster.")
    movie_rate: str = Field(..., description="Your rating of the movie.")
    movie_review: str = Field(..., description="Your review of the movie.")

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.fireworks.ai/inference/v1", 
        api_key=getenv("FIREWORKS_API_KEY")
    ),
    mode=instructor.Mode.JSON
)

result = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
    response_model=MovieRate,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a picture about a movie. Please figure out the movie name, rate the movie and review comment."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://mickeyblog.com/wp-content/uploads/2018/11/2018-11-05-20_41_02-Toy-Story-4_-Trailer-Story-Cast-Every-Update-You-Need-To-Know-720x340.png"
                    }
                }
            ]
        }
    ],
)
print(result)

#### autogen with Llama3.2
!pip install pyautogen

import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from types import SimpleNamespace
import instructor
from openai import OpenAI
from os import getenv

import re

class TravelDestination(BaseModel):
    destination_name: str = Field(..., description="The name of the destination.")
    attractions: Optional[List[str]] = Field(..., description="Attractions at the destination.")
    transportation: Optional[List[str]] = Field(..., description="Transportation options to the destination.")
    accommodation: Optional[List[str]] = Field(..., description="Hotels near the destination.")
    restaurants: Optional[List[str]] = Field(..., description="Restaurants near the destination.")
    description: Optional[str] = Field(..., description="Description of the destination.")

class CustomLLama32VisionClient:
    def __init__(self, config, **kwargs):
        print(f"CustomLLMClient config: {config}")

    def create(self, params):

        new_messages = []
        for message in params["messages"]:
            if message["role"] == "user": 
                new_content = []
                text_content = ""
                for match in re.finditer(r"<url>(.*?)</url>", message["content"]):
                    url = match.group(1)
                    new_content.append({"type": "image_url", "image_url": {"url": url}})
                    text_content = message["content"].replace(f"<url>{url}</url>", "").strip()

                if text_content:
                    new_content.insert(0, {"type": "text", "text": text_content})

                new_messages.append({"role": "user", "content": new_content})

        client = instructor.from_openai(OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=getenv("FIREWORKS_API_KEY")), mode=instructor.Mode.JSON)
        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            messages=new_messages,
            response_model=TravelDestination, 
        )

        autogen_response = SimpleNamespace()
        autogen_response.choices = []
        autogen_response.model = "custom_llama32_vision"  

        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = response.model_dump_json()
        choice.message.function_call = None
        autogen_response.choices.append(choice)
        return autogen_response

    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        response.cost = 0 
        return 0

    @staticmethod
    def get_usage(response):
        return {}

config_list=[
    {
        "model": "custom_llama32_vision",
        "model_client_cls": "CustomLLama32VisionClient",
    }
]

user_proxy = UserProxyAgent("user_proxy", code_execution_config=False, human_input_mode="TERMINATE")
vision_assistant = AssistantAgent(
    "vision_assistant",
    system_message="You are a multi-modal assistant that can generate structured outputs from images.",
    llm_config={
        "config_list": config_list,  
        "cache": None, 
        "cache_seed": None
    },
)
vision_assistant.register_model_client(model_client_cls=CustomLLama32VisionClient)

config_list_text = [{"model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct", "api_key": getenv("FIREWORKS_API_KEY"), "base_url": "https://api.fireworks.ai/inference/v1"}]
blog_assistant = AssistantAgent(
    "blog_assistant",
    system_message="You are an assistant that can generate blog posts based on the given information. Terminate your response with TERMINATE.",
    llm_config={
        "config_list": config_list_text,  
        "cache": None, 
        "cache_seed": None
    },
)

groupchat = autogen.GroupChat(agents=[user_proxy, vision_assistant, blog_assistant], messages=[], max_round=6)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_text})

result = user_proxy.initiate_chat(
    manager,
    message="""
          This is a picture of a famous destination. 
          Please figure out the destination name, attractions, transportation, accommodation, food, and description.
          And then generate a blog post based on the given information.
          <url>https://cdn.britannica.com/89/179589-138-3EE27C94/Overview-Great-Wall-of-China.jpg?w=800&h=450&c=crop</url>
            """

)
print(result)
