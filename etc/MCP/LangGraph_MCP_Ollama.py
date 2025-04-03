### From https://medium.com/data-science-collective/langgraph-mcp-ollama-the-key-to-powerful-agentic-ai-e1881f43cf63

!pip install -r requirements.txt

# agent.py
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from nodes import create_chatbot
import asyncio
import os
import dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
# main.py
import streamlit as st
import asyncio
from agent import create_agent
from langchain_core.messages import HumanMessage
# nodes.py
from server import get_tools
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from datetime import datetime
import os

# server.py - unchanged except for removing search_google if it's not working
from mcp.server.fastmcp import FastMCP
from langchain_experimental.utilities import PythonREPL
import io
import base64
import matplotlib.pyplot as plt
from openai import OpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import asyncio
from googlesearch import search  

async def create_agent(docs_info=None):
    async with MultiServerMCPClient(
        {
            "server":{
                "url":"http://localhost:8000/sse",
                "transport":"sse",
                "timeout": 30
            }
        }
    ) as client:
        # Get MCP tools
        tools = client.get_tools()
        
        # Create the graph builder
        graph_builder = StateGraph(MessagesState)
        
        # Create nodes
        chatbot_node = create_chatbot(docs_info)
        graph_builder.add_node("chatbot", chatbot_node)

      
  # Custom async tool node to handle async MCP tools
        async def async_tool_executor(state):
            messages = state["messages"]
            last_message = messages[-1]
            
            # Check if there are tool calls
            tool_calls = None
            if hasattr(last_message, "tool_calls"):
                tool_calls = last_message.tool_calls
            elif hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs["tool_calls"]
                
            if not tool_calls:
                return {"messages": messages}
            
            # Process each tool call
            new_messages = messages.copy()
            
            for tool_call in tool_calls:
                # Handle different formats of tool_call
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "tool-call-id")
                else:
                    tool_name = tool_call.name
                    tool_args = tool_call.args if hasattr(tool_call, "args") else {}
                    tool_id = getattr(tool_call, "id", "tool-call-id")
                
                # Print debug info
                print(f"Executing tool: {tool_name}")
                print(f"Tool args: {tool_args}")
                
                # Find the matching tool
                tool = next((t for t in tools if t.name == tool_name), None)
                
                if not tool:
                    # Tool not found
                    tool_error = f"Error: {tool_name} is not a valid tool, try one of {[t.name for t in tools]}."
                    new_messages.append(AIMessage(content=tool_error))
                else:
                    try:
                        # Execute the async tool
                        if asyncio.iscoroutinefunction(tool.coroutine):
                            result = await tool.coroutine(**tool_args)
                        else:
                            # Fall back to sync execution if needed
                            result = tool.func(**tool_args) if hasattr(tool, 'func') else tool(**tool_args)
                        
                        print(f"Tool result: {result}")
                        
                        # Add tool result
                        new_messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                            name=tool_name
                        ))
                    except Exception as e:
                        # Handle errors
                        error_msg = f"Error: {str(e)}\n Please fix your mistakes."
                        print(f"Tool error: {error_msg}")
                        new_messages.append(AIMessage(content=error_msg))
            
            return {"messages": new_messages}

      
 # Add the async tool executor node
        graph_builder.add_node("tools", async_tool_executor)
        
        # Define router function to handle tool calls
        def router(state):
            messages = state["messages"]
            last_message = messages[-1]
            
            has_tool_calls = False
            if isinstance(last_message, AIMessage):
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    has_tool_calls = True
                elif hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("tool_calls"):
                    has_tool_calls = True
            
            return "tools" if has_tool_calls else "end"
        
        # Add edges
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            router,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph_builder.add_edge("tools", "chatbot")
        
        # Compile the graph
        graph = graph_builder.compile()
        return graph, client  # Return client to keep it alive


def get_system_prompt(docs_info=None):
    system_prompt = f"""
    Today is {datetime.now().strftime("%Y-%m-%d")}
    You are a helpful AI Assistant that can use these tools:
    - generate_image: Generate an image using DALL-E based on a prompt
    - data_visualization: Create charts with Python and matplotlib
    - python_repl: Execute Python code
    
    When you call image generation or data visualization tool, only answer the fact that you generated, not base64 code or url.
    Once you generated image by a tool, then do not call it again in one answer.
    """
    if docs_info:
        docs_context = "\n\nYou have access to these documents:\n"
        for doc in docs_info:
            docs_context += f"- {doc['name']}: {doc['type']}\n"
        system_prompt += docs_context
        
    system_prompt += "\nYou should always answer in same language as user's ask."
    return system_prompt

def create_chatbot(docs_info=None):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(get_system_prompt(docs_info)),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Use the LLM without binding tools
    chain = prompt | llm
    
    def chatbot(state: MessagesState):
        # Ensure messages are in the right format
        if isinstance(state["messages"], str):
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=state["messages"])]
        else:
            messages = state["messages"]
            
        response = chain.invoke(messages)
        return {"messages": messages + [response]}
    
    return chatbot


@mcp.tool()
async def generate_image(prompt: str) -> str:
    """
    Generate an image using DALL-E based on the given prompt.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Invalid prompt")
    
    try:
        # Since this is an async function, we need to handle the synchronous OpenAI call properly
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
        )
        
        # Return both success message and URL
        return f"Successfully generated an image of {prompt}! Here's the URL: {response.data[0].url}"
    except Exception as e:
        return f"Error generating image: {str(e)}"

repl = PythonREPL()

@mcp.tool()
def data_visualization(code: str):
    """Execute Python code. Use matplotlib for visualization."""
    try:
        repl.run(code)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"Error creating chart: {str(e)}"
        
@mcp.tool()
def python_repl(code: str):
    """Execute Python code."""
    return repl.run(code)

def get_tools(retriever_tool=None):
    # Only include tools that are working
    base_tools = [generate_image, python_repl, data_visualization]
    
    if retriever_tool:
        base_tools.append(retriever_tool)
    
    return base_tools

if __name__ == "__main__":
    mcp.run(transport="sse")


async def main():
    # Create the agent
    agent, client = await create_agent()
    
    # Get user input from command line
    user_input = input("What would you like to ask? ")
    
    # Create a proper initial message
    initial_message = HumanMessage(content=user_input)
    
    try:
        # Use the agent asynchronously
        print("Processing your request...")
        result = await agent.ainvoke({"messages": [initial_message]})
        
        # Print the results
        for message in result["messages"]:
            if hasattr(message, "type") and message.type == "human":
                print(f"User: {message.content}")
            elif hasattr(message, "type") and message.type == "tool":
                print(f"Tool Result: {message.content}")
                # If it's an image generation result, extract URL
                if "image" in message.content.lower() and "url" in message.content.lower():
                    print("Image Generated Successfully!")
            else:
                print(f"AI: {message.content}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Keep the client alive until all operations are done
    # In a real application, you'd keep the client active as long as needed

if __name__ == "__main__":
    asyncio.run(main())
