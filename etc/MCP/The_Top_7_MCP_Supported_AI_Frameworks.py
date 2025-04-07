### From https://medium.com/@amosgyamfi/the-top-7-mcp-supported-ai-frameworks-a8e5030c87ab
"""
1. LLM Context Provision & Limitations
   -a. Basic Limitation:
       By default, LLMs and AI chatbots lack a proper context, which prevents them from fetching real-time information, 
       executing code, calling external tools and APIs, or using a web browser on behalf of users.
   -b. Developer Solutions:
       Developers can overcome these limitations using several approaches to provide the necessary context and enable external tool
       integration.

2. Approaches and Specifications
   -a. Composio:
       -1. Offers specifications and a comprehensive library of toolkits for integrating AI agents and LLMs.
       -2. Recently introduced Composio MCP, which allows developers to connect to over 100 MCP servers within IDEs such as Cursor, 
           Claude, and Windsurf.
   -b. Agents.json:
       -1. A specification built on OpenAI standards to facilitate seamless interaction between AI agents and external APIs or tools.
       -2. Although well-designed, it is less widely adopted compared to MCP.
   -c. MCP (Modular Context Provision):
       -1. Provides the best method for supplying contextual data to LLMs and AI assistants to solve problems.
       -2. For example, you can set up an MCP documentation server that grants IDEs and agentic frameworks complete access to your 
           documentation—similar to using an llms.txt file.

3. What is MCP?
   -a. Evolution of LLMs:
       -1. First Evolution: LLMs answer user prompts accurately only if the queries exist in their training data; 
                            they lack external tool access.
       -2. Second Evolution: LLMs are given access to additional context via external tools, though these tools can be unwieldy 
                             to use.
       -3. Third Evolution (MCP): Implements a robust infrastructure to allow LLMs and their tools to access external applications 
                                  in a maintainable and standardized way.
   -b. Enterprise and AI-Assisted Coding:
       -1. MCP connects enterprise cloud data (e.g., customer support ticket systems) to AI systems.
       -2. It offers a standardized method to integrate content repositories (like GitHub and Notion), development environments, 
           web resources, and business tools with AI technologies.
       -3. A prominent use case is AI-assisted coding, with hundreds of integrations available for development tools.

4. How MCP Works
   -a. Use Case Example:
       Without MCP, asking ChatGPT to message a Slack channel, check calendar availability, or schedule a meeting will likely result
       in disappointment because it lacks access to those applications.
   -b. Operational Flow:
       -1. A user sends a query to an agent.
       -2. The agent determines which MCP server and tool to call to fetch the necessary data.
       -3. The agent then uses the received data from the specific tool to generate a meaningful response.

5. Why Adopt MCP for AI Agents and LLM-Based Apps?
   -a. Standardization and Integration:
       MCP is emerging as a standard that helps AI systems communicate effectively with external applications.
       -1. Microsoft has integrated MCP in its Copilot Studio to streamline tool access.
       -2. OpenAI supports MCP in products like the Agents SDK and the ChatGPT desktop app.
   -b. Managing Complexity:
       When AI agents perform multiple tasks (e.g., reading emails, web scraping, financial analysis, fetching real-time weather), 
       directly integrating each tool becomes cumbersome.
       MCP centralizes tool management, allowing even hundreds of tools to be accessed via a central registry.

6. Advantages of MCP Over Traditional Tool Integration
   -a. Architecture:
       MCP features a clean, flexible architecture for interacting with tools and APIs.
   -b. Improved Tool Access & Management:
       It offers a standardized interface for LLMs to interact with third-party systems, reducing errors and compatibility issues.
   -c. Reduced Development Time:
       Traditional methods require custom code for each tool, a process that can take weeks per integration. 
       MCP standardizes these processes.
   -d. Key Benefits:
       -1. Enhanced Authentication: Robust built-in authentication and permissions 
           (e.g., authenticating users with Google Sheets or Gmail via Composio-provided MCP tools).
       -2. Ease of Tool Search: Simplifies finding and integrating external tools.
       -3. Scalability: Supports many users and applications.
       -4. Community-Driven: Widely adopted open-source servers and a vibrant developer ecosystem.
       -5. Industry Standardization: Establishes a standard for providing the required context to AI agents and LLMs.

7. Types of MCP Servers
   -a. Server-Sent Events (SSE):
       Connects to remote services via HTTP.
   -b. STDIO:
       Enables the execution of local commands and communication via standard I/O.
   -c. Framework Support:
       The chosen AI development framework typically provides the necessary classes to interface with these server types.

8. Ecosystem of MCP Registries/Servers
   -a. Hosted MCP Tools:
       Several open-source libraries of hosted MCP tools (called registries) offer curated collections of services that enhance LLM 
       and agent functionality.
   -b. Available Registries and Options:
       -1. MCP servers on GitHub: Community-built servers with additional MCP resources.
       -2. Glama Registry: Production-ready, open-source MCP servers for developers.
       -3. Smithery Registry: Provides access to over 2000 MCP servers to extend AI agent capabilities.
   -c. OpenTools:
       -1. Offers generative APIs for MCP tool use, allowing developers to expand LLM capabilities
           (e.g., web search, fetching real-time location data, web scraping).
       -2. The API supports Curl, Python, and TypeScript.
       -3. Code example included below:
           """
           from openai import OpenAI

           client = OpenAI(
               base_url="https://api.opentools.com",
               api_key="<OPENTOOLS_API_KEY>"
           )

           completion = client.chat.completions.create(
               model="anthropic/claude-3.7-sonnet",
               messages=[
                   { "role": "user", "content": "Compare specs of top 5 EVs on caranddriver.com" }
               ],
               tools=[{ "type": "mcp", "ref": "firecrawl" }]
           )
           """"
   -d. PulseMCP Registry:
       Allows browsing of hosted MCP tools and trending use cases for AI projects.
   -e. mcp.run:
       Gives developers access to hundreds of MCP apps for business purposes.
   -f. Composio Registry:
       Uses SSE-based MCP servers for seamless integration with various AI frameworks.
   -g. guMCP:
       Provides free, open-source, fully hosted MCP servers that integrate seamlessly with any AI app.
""""
### Top 7 Client Frameworks to Add MCP to LLMs and Agents
##  1. Build a Git MCP Agent With OpenAI Agents SDK
import asyncio
import shutil
import streamlit as st
from agents import Agent, Runner, trace
from agents.mcp import MCPServer, MCPServerStdio

async def query_git_repo(mcp_server: MCPServer, directory_path: str, query: str):
    agent = Agent(
        name="Assistant",
        instructions=f"Answer questions about the localgit repository at {directory_path}, use that for repo_path",
        mcp_servers=[mcp_server],
    )

    with st.spinner(f"Running query: {query}"):
        result = await Runner.run(starting_agent=agent, input=query)
        return result.final_output

async def run_streamlit_app():
    st.title("Local Git Repo Explorer")
    st.write("This app allows you to query information about a local git repository.")

    directory_path = st.text_input("Enter the path to the git repository:")

    if directory_path:
        # Common queries as buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Most frequent contributor"):
                query = "Who's the most frequent contributor?"
                run_query(directory_path, query)

        with col2:
            if st.button("Last change summary"):
                query = "Summarize the last change in the repository."
                run_query(directory_path, query)

        # Custom query
        custom_query = st.text_input("Or enter your own query:")
        if st.button("Run Custom Query") and custom_query:
            run_query(directory_path, custom_query)

def run_query(directory_path, query):
    if not shutil.which("uvx"):
        st.error("uvx is not installed. Please install it with `pip install uvx`.")
        return

    async def execute_query():
        async with MCPServerStdio(
            cache_tools_list=True,
            params={
                "command": "python", 
                "args": [
                    "-m", 
                    "mcp_server_git", 
                    "--repository", 
                    directory_path
                ]
            },
        ) as server:
            with trace(workflow_name="MCP Git Query"):
                result = await query_git_repo(server, directory_path, query)
                st.markdown("### Result")
                st.write(result)

    asyncio.run(execute_query())

if __name__ == "__main__":
    st.set_page_config(
        page_title="Local Git Repo Explorer",
        page_icon="📊",
        layout="centered"
    )
    # Change from async to synchronous implementation
    # Since Streamlit doesn't work well with asyncio in the main thread

    # Define a synchronous version of our app
    def main_streamlit_app():
        st.title("Local Git Repo Explorer")
        st.write("This app allows you to query information about a Git repository.")

        directory_path = st.text_input("Enter the path to the git repository:")

        if directory_path:
            # Common queries as buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Most frequent contributor"):
                    query = "Who's the most frequent contributor?"
                    run_query(directory_path, query)

            with col2:
                if st.button("Last change summary"):
                    query = "Summarize the last change in the repository."
                    run_query(directory_path, query)

            # Custom query
            custom_query = st.text_input("Or enter your own query:")
            if st.button("Run Custom Query") and custom_query:
                run_query(directory_path, custom_query)

    # Run the synchronous app
    main_streamlit_app()
--------------------------------------------------------------------------------
## 2. Build MCP AI Agents With Praison AI
import streamlit as st
from praisonaiagents import Agent, MCP

st.title("🏠 Airbnb Booking Assistant")

# Create the agent
@st.cache_resource
def get_agent():
    return Agent(
        instructions="""You help book apartments on Airbnb.""",
        llm="gpt-4o-mini",
        tools=MCP("npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt")
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input form
with st.form("booking_form"):
    st.subheader("Enter your booking details")

    destination = st.text_input("Destination:", "Paris")

    col1, col2 = st.columns(2)
    with col1:
        check_in = st.date_input("Check-in date")
    with col2:
        check_out = st.date_input("Check-out date")

    adults = st.number_input("Number of adults:", min_value=1, max_value=10, value=2)

    submitted = st.form_submit_button("Search for accommodations")

    if submitted:
        search_agent = get_agent()

        # Format the query
        query = f"I want to book an apartment in {destination} from {check_in.strftime('%m/%d/%Y')} to {check_out.strftime('%m/%d/%Y')} for {adults} adults"

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Get response from the agent
        with st.chat_message("assistant"):
            with st.spinner("Searching for accommodations..."):
                response = search_agent.start(query)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Allow for follow-up questions
if st.session_state.messages:
    prompt = st.chat_input("Ask a follow-up question about the accommodations")
    if prompt:
        search_agent = get_agent()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = search_agent.start(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 
--------------------------------------------------------------------------------
## 3. Using MCP for LangChain AI Apps
# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import asyncio
import pathlib
import sys
import typing as t

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp import MCPToolkit

async def run(tools: list[BaseTool], prompt: str) -> str:
    model = ChatGroq(model_name="llama-3.1-8b-instant", stop_sequences=None)  # requires GROQ_API_KEY
    tools_map = {tool.name: tool for tool in tools}
    tools_model = model.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(prompt)]
    ai_message = t.cast(AIMessage, await tools_model.ainvoke(messages))
    messages.append(ai_message)
    for tool_call in ai_message.tool_calls:
        selected_tool = tools_map[tool_call["name"].lower()]
        tool_msg = await selected_tool.ainvoke(tool_call)
        messages.append(tool_msg)
    return await (tools_model | StrOutputParser()).ainvoke(messages)

async def main(prompt: str) -> None:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(__file__).parent.parent)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            toolkit = MCPToolkit(session=session)
            await toolkit.initialize()
            response = await run(toolkit.get_tools(), prompt)
            print(response)

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Read and summarize the file ./readme.md"
    asyncio.run(main(prompt))
--------------------------------------------------------------------------------
## 4. Using MCP for Chainlit AI Apps
# pip install chainlit

import chainlit as cl
from mcp import ClientSession

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    # Your connection initialization code here
    # This handler is required for MCP to work

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Optional handler: Cleanup your code here
--------------------------------------------------------------------------------
## 5. Integrate MCP for Agno AI Agents
    # Define server parameters
    airbnb_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        env=env,
    )

    maps_server_params = StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-google-maps"], env=env
    )

    # Use contextlib.AsyncExitStack to manage multiple async context managers
    async with contextlib.AsyncExitStack() as stack:
        # Create stdio clients for each server
        airbnb_client, _ = await stack.enter_async_context(stdio_client(airbnb_server_params))
        maps_client, _ = await stack.enter_async_context(stdio_client(maps_server_params))

        # Create all agents
        airbnb_agent = Agent(
            name="Airbnb",
            role="Airbnb Agent",
            model=OpenAIChat("gpt-4o"),
            tools=[airbnb_client],
            instructions=dedent("""\
                You are an agent that can find Airbnb listings for a given location.\
            """),
            add_datetime_to_instructions=True,
        )
--------------------------------------------------------------------------------
## 6. Using MCP for Upsonic Agents
import os
from dotenv import load_dotenv
from upsonic import Task, Agent, Direct
from upsonic.client.tools import Search  # Adding Search as a fallback tool

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Set your OpenAI API key for the session
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define the HackerNews MCP tool
# Using the correct MCP setup for HackerNews based on Upsonic documentation
class HackerNewsMCP:
    command = "uvx"
    args = ["mcp-hn"]
    # No environment variables are needed for this MCP

# Create a task to analyze the latest HackerNews stories
# Adding Search as a fallback in case HackerNews MCP fails
task = Task(
    "Analyze the top 5 HackerNews stories for today. Provide a brief summary of each story, "
    "identify any common themes or trends, and highlight which stories might be most relevant "
    "for someone interested in AI and software development.",
    tools=[HackerNewsMCP, Search]  # Include both HackerNews MCP and Search tools
)

# Create an agent specialized in tech news analysis
agent = Agent(
    "Tech News Analyst",
    company_url="https://news.ycombinator.com/",
    company_objective="To provide insightful analysis of tech industry news and trends"
)

# Execute the task with the agent and print the results
print("Analyzing HackerNews stories...")
agent.print_do(task)

# Alternatively, you can use a Direct LLM call if the task is straightforward
# print("Direct analysis of HackerNews stories...")
# Direct.print_do(task)

# If you want to access the response programmatically:
# agent.do(task)
# result = task.response
# print(result)
--------------------------------------------------------------------------------
## 7. Using MCP for Mastra Agents
import { MCPConfiguration } from "@mastra/mcp";
import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";

const mcp = new MCPConfiguration({
  servers: {
    stockPrice: {
      command: "npx",
      args: ["tsx", "stock-price.ts"],
      env: {
        API_KEY: "your-api-key",
      },
    },
    weather: {
      url: new URL("http://localhost:8080/sse"),
    },
  },
});

// Create an agent with access to all tools
const agent = new Agent({
  name: "Multi-tool Agent",
  instructions: "You have access to multiple tool servers.",
  model: openai("gpt-4"),
  tools: await mcp.getTools(),
});



