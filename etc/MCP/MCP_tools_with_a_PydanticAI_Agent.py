### From https://medium.com/@finndersen/how-to-use-mcp-tools-with-a-pydanticai-agent-0d3a09c93a51

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic_ai import Agent
from rich.console import Console
from rich.prompt import Prompt

# Define server configuration
server_params = StdioServerParameters(
      command="npx",
      args=[
          "tsx",
          "server/index.ts",
          "path/to/working/directory",
      ],
  )

# Start server and connect client
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()
        
        # Get details of available tools
        tools_result = await session.list_tools()
        tools = tools_result.tools

async def get_tools(session: ClientSession) -> list[Tool[AgentDeps]]:
    """
    Get all tools from the MCP session and convert them to Pydantic AI tools.
    """
    tools_result = await session.list_tools()
    return [pydantic_tool_from_mcp_tool(session, tool) for tool in tools_result.tools]


def pydantic_tool_from_mcp_tool(session: ClientSession, tool: MCPTool) -> Tool[AgentDeps]:
    """
    Convert a MCP tool to a Pydantic AI tool.
    """
    tool_function = create_function_from_schema(session=session, name=tool.name, schema=tool.inputSchema)
    return Tool(name=tool.name, description=tool.description, function=tool_function, takes_ctx=True)

def create_function_from_schema(session: ClientSession, name: str, schema: Dict[str, Any]) -> types.FunctionType:
    """
    Create a function with a signature based on a JSON schema. This is necessary because PydanticAI does not yet
    support providing a tool JSON schema directly.
    """
    # Create parameter list from tool schema
    parameters = convert_schema_to_params(schema)

    # Create the signature
    sig = inspect.Signature(parameters=parameters)

    # Create function body
    async def function_body(ctx: RunContext[AgentDeps], **kwargs) -> str:
        # Call the MCP tool with provided arguments
        result = await session.call_tool(name, arguments=kwargs)

        # Assume response is always TextContent
        if isinstance(result.content[0], TextContent):
            return result.content[0].text
        else:
            raise ValueError("Expected TextContent, got ", type(result.content[0]))

    # Create the function with the correct signature
    dynamic_function = types.FunctionType(
        function_body.__code__,
        function_body.__globals__,
        name=name,
        argdefs=function_body.__defaults__,
        closure=function_body.__closure__,
    )

    # Add signature and annotations
    dynamic_function.__signature__ = sig  # type: ignore
    dynamic_function.__annotations__ = {param.name: param.annotation for param in parameters}

    return dynamic_function

def run():  
    # Configure Model and Agent dependencies
    ...
    # Initialise & connect MCP server, construct tools
    ...

    agent = Agent(
        model=model,
        deps_type=type(deps),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )

    message_history: list[ModelMessage] = []
    while True:
        prompt = Prompt.ask("[cyan]>[/cyan] ").strip()

        if not prompt:
            continue

        # Handle special commands
        if prompt.lower() in EXIT_COMMANDS:
            break

        # Process normal input through the agent
        result = await agent.run(prompt, deps=deps, message_history=message_history)
        response = result.data

        console.print(f"[bold green]Agent:[/bold green] {response}")
        message_history = result.all_messages()
