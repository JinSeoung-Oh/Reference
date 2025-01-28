### https://medium.com/@samarrana407/autogen-0-4-build-smarter-ai-agents-with-ease-cae182506ea2

"""
1. Introduction to Autogen
   -a. Evolution of Autogen
       -1. Autogen began as an innovative agentic framework designed to create autonomous software agents driven by large language models (LLMs).
       -2. Over time, it has transitioned into a modular and extensible platform, reaching a new milestone with version 0.4.
   -b. Purpose of Autogen
       -1. Autogen simplifies the development of LLM-powered agents capable of performing a broad range of tasks.
       -2. Agents respond to text queries and can integrate with external APIs, coordinate amongst themselves, and maintain state across complex, 
           long-running sessions.

2. What Is Autogen?
   -a. Agentic Framework Overview
       -1. Autogen provides a framework for building “agents”: autonomous pieces of software that leverage LLMs.
       -2. These agents accept simple text-based queries and perform sophisticated tasks such as:
           -1) Web searches to fetch information.
           -2) Summarizing or organizing large volumes of data.
           -3) Coordinating with other agents to accomplish multi-step tasks.
   -b. Key Capabilities
       -a. Real-Time Information Access: Agents can connect to external APIs for up-to-date data.
       -b. Multiple-Agent Coordination: The framework allows multiple agents, each with its own role and capabilities, to collaborate.
       -c. Event-Driven Architecture: Agents can proactively run in long-lived or asynchronous modes, reacting to various events in real time.

3. Autogen 0.4: Key Features and Updates
   Autogen 0.4 introduces major enhancements that further solidify its position as a cutting-edge, agentic development platform:

   -a. Asynchronous Messaging
       -1. Agents now communicate through asynchronous messages, accommodating event-driven and request/response patterns.
       -2. This update offers better scalability and flexibility for complex workflows.
   -b. Modular and Extensible Architecture
       -1. Provides a pluggable system of components like custom agents, tools, memory, and models.
       -2. Simplifies building proactive, long-running agents that can be extended or customized for various use cases.
   -c. Full Type Support
       -1. Ensures consistent interfaces and strict typing across the codebase.
       -2. Enforces high-quality code and more reliable APIs.
   -d. Layered Architecture
       -1. Allows developers to pick the abstraction level that best suits their scenario, from high-level orchestration to low-level agent behaviors.
   -e. Observability and Debugging
       -1. Offers built-in tools for tracking, tracing, and debugging agent interactions and workflows.
       -2. OpenTelemetry integration for better insight into agent performance and state changes.
   -f. Scalable and Distributed
       -1. Enables the creation of distributed agent networks that can operate smoothly across different organizational boundaries.
       -2. Designed for environments requiring high availability and fault-tolerance.
   -g. Cross-Language Support
       -1. Facilitates interoperability between agents built in different programming languages, currently including Python and .NET.
   -h.Built-in and Community Extensions
      -1. The framework supports community-driven extensions for advanced or specialized functionalities.
      -2. Encourages open-source collaboration where developers can create and maintain their own extensions.
"""
conda create -n autogen python==3.12 -y
conda activate autogen
pip install -U "autogen-agentchat" "autogen-ext[openai]"

export OPENAI_API_KEY="your_api_key_here"

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    agent = AssistantAgent("assistant", OpenAIChatCompletionClient(model="gpt-4o"))
    print(await agent.run(task="Say 'Hello World!'"))

asyncio.run(main())












  
