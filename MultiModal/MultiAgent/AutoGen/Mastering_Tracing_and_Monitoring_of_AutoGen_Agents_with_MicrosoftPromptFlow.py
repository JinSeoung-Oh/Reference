### From https://pub.towardsai.net/mastering-tracing-and-monitoring-of-autogen-agents-with-microsoft-promptflow-9b9454d98734

"""
1. Overview of the Blog Content
   -a. Purpose of the Blog:
       The blog is written to explain the basics of AI agents, the concept of an agentic approach, 
       and the various types of agents. It invites readers to revisit these fundamentals.

2. Microsoft Promptflow
   -a. Description:
       Microsoft Promptflow is introduced as a comprehensive development toolkit for building AI applications 
       based on large language models (LLMs).
   -b. Capabilities and Benefits:
       -1. End-to-End Lifecycle Support:
           Simplifies the entire process from concept and prototyping to testing, evaluation, deployment, and monitoring.
       -2. Enhanced Prompt Engineering:
           Streamlines the creation of production-ready LLM applications.
       -3. Core Functions:
           -1) Create and iteratively develop flows.
           -2) Evaluate flow quality and performance.
           -3) Streamline the development cycle for production.
           -4) Offers additional functionalities to support the prompt development process.

3. Microsoft Autogen
   -a. Description:
       Microsoft Autogen is presented as a framework designed to facilitate the creation, management, 
       and interaction of AI agents, particularly in the context of LLMs.
   -b. Key Features:
       -1. Agent Creation and Management:
           Enables the development of intelligent agents that operate autonomously.
       -2. Multi-Agent Orchestration:
           Supports the creation of multiple agents with distinct roles (e.g., user proxy, assistant, helper) 
           and manages their communication, task execution, and tracing.
       -3. Application Context:
           Useful in scenarios requiring the orchestration of several AI agents to tackle complex tasks or automate processes.

4. Monitoring and Evaluation of Agentic Workflows
   -a. Importance of Tracking and Tracing:
       -1. Emphasizes that tracking, monitoring, and evaluating agent interactions is crucial.
       -2. Without visibility into the conversation flow or output, it is challenging to determine which prompts, code segments, or parameters need tweaking.
   -b. Output Display Methods:
       -1. Printing:
           -1) Can be used to display output, but may lead to issues with formatting and organization.
       -2. Logging:
           -1) Offers a more structured method for recording output.
           -2) Challenges include the potential for log files to become excessively large, performance issues, 
               and difficulties in managing and extracting meaningful information.
           -3) Effective log rotation and management strategies are necessary to handle these challenges.
   -c. Role of Promptflow:
       -1. The blog mentions that Promptflow provides a way to monitor agentic workflows.
       -2. Although there are issues that one might encounter, the blog also includes solutions for these common problems.
"""
------------------------------------------------------------------------------------------------
import autogen
from autogen import AssistantAgent
import os
import json
import logging
from autogen.retrieve_utils import TEXT_FORMATS
from promptflow.tracing import start_trace
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

# Set up logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Configuration
env_or_file = "OAI_CONFIG_LIST.json"
config_list = autogen.config_list_from_json(
    env_or_file,
    filter_dict={
        "model": {
            "gpt-35-turbo",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
            "gpt-4o-mini"
        },
    },
)

llm_config = {"config_list": config_list, "cache_seed": 42}
#assert len(config_list) > 0
print("Models to use: ", [config["model"] for config in config_list])

""" env_or_file
[
    {
        "model": "gpt-4o-mini",
        "api_key": "Your OpenAI api key",
        "timeout": 600


    }
]
"""

# Agents

# Simple user proxy agent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },
    human_input_mode="TERMINATE"
)

# Conversable agent 1
assistant = autogen.ConversableAgent(
    name="assistant",
    system_message="Create a story on a given topic.",
    llm_config=llm_config,

)

# # Conversable agent 2
helper = autogen.ConversableAgent(
    name="helper",
    system_message="Check for tone, jump scares, and other factors to make the story interesting.",
    llm_config=llm_config,
)

# Conversable agent 3
helper2 = autogen.ConversableAgent(
        name="helper2",
        system_message="Review the story and give suggestion to make it stunning..",
        llm_config=llm_config,

)

# GroupChat setup
groupchat = autogen.GroupChat(agents=[assistant, helper, helper2], messages=[], max_round=12,speaker_selection_method="round_robin")
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Create a root span for the interaction
with tracer.start_as_current_span("autogen-main") as main_span:
    try:
        message = "Write a story about stars and explain it "
        user_proxy.initiate_chat(
            manager,
            message=message,
        )
        main_span.set_attribute("custom", "custom attribute value")
        main_span.add_event(
            "promptflow.function.inputs", {"payload": json.dumps(dict(message=message))}
        )

        # Trace each agent's message exchange
        for idx, interaction in enumerate(groupchat.messages):
            with tracer.start_as_current_span(f"interaction-{idx}") as span:
                try:
                    if isinstance(interaction, dict):
                        # Extract sender and message information from the dictionary
                        sender = interaction.get("sender", {}).get("name", "Unknown Sender")
                        message_content = interaction.get("content", "No content")
                    else:
                        sender = getattr(interaction.sender, "name", "Unknown Sender")
                        message_content = getattr(interaction, "content", "No content")

                    span.set_attribute("agent", sender)
                    span.set_attribute("message", message_content)
                    span.add_event("agent.message_received", {
                        "sender": sender,
                        "message": message_content,
                    })
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    logger.error(f"Error in interaction-{idx}: {e}")
                    span.set_status(Status(StatusCode.ERROR, description=str(e)))
                    span.add_event("error", {"exception": str(e)})

        main_span.set_status(Status(StatusCode.OK))
        main_span.add_event(
            "promptflow.function.output", {"payload": json.dumps(user_proxy.last_message())}
        )
    except Exception as e:
        logger.error(f"Error in main interaction: {e}")
        main_span.set_status(Status(StatusCode.ERROR, description=str(e)))
        main_span.add_event("error", {"exception": str(e)})

