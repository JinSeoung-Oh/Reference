### From https://medium.com/google-cloud/model-context-protocol-mcp-with-google-gemini-llm-a-deep-dive-full-code-ea16e3fac9a3

#Setup virtual env
python -n venv venv 

#Activate venv
source venv/bin/activate

#Install dependancies
pip install google-genai mcp

export GEMINI_API_KEY="your-google-api-key"
export SERP_API_KEY="your-serpapi-key"

# Install from PyPI
pip install mcp-flight-search

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

server_params = StdioServerParameters(
    command="mcp-flight-search",
    args=["--connection_type", "stdio"],
    env={"SERP_API_KEY": os.getenv("SERP_API_KEY")},
)

async def run():
    # Remove debug prints
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            prompt = f"Find Flights from Atlanta to Las Vegas 2025-05-05"
            await session.initialize()
            # Remove debug prints

            mcp_tools = await session.list_tools()
            # Remove debug prints
            tools = [
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in mcp_tools.tools
            ]
            # Remove debug prints

            response = client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=tools,
                ),
            )


response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25",
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0,
        tools=tools,
    ),
)

result = await session.call_tool(
    function_call.name, arguments=dict(function_call.args)
)


