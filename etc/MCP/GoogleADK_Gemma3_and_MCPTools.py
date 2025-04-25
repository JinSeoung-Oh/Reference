### From https://medium.com/google-cloud/building-ai-agents-with-google-adk-gemma-3-and-mcp-tools-28763a8f3c62

# Setup virtual environment (Mac or Unix )
python -m venv venv && source venv/bin/active 

# Install agent development kit
pip install google-adk

# Install MCP server 
pip install mcp-youtube-search

# Install litellm
pip install litellm

pip install mcp-youtube-search

export SERP_API_KEY="your-serpapi-key"

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm
import os
import dotenv

# Step 2: Load environment variables
dotenv.load_dotenv()

# Step 3: Create tool execution helper functions
async def execute_tool(tool, args):
    """Execute a single tool and handle cleanup."""
    try:
        result = await tool.run_async(args=args, tool_context=None)
        return (True, result, None)  # Success, result, no error
    except Exception as e:
        return (False, None, str(e))  # Failed, no result, error message


async def try_tools_sequentially(tools, args, exit_stack):
    """Try each tool in sequence until one succeeds."""
    errors = []
    
    for tool in tools:
        success, result, error = await execute_tool(tool, args)
        if success:
            return result
        errors.append(f"Tool '{tool.name}' failed: {error}")
    
    if errors:
        return f"All tools failed: {'; '.join(errors)}"
    return "No tools available"

# Step 4: Create MCP tool executor
def create_mcp_tool_executor(command, args=None, env=None):
    """Create a function that connects to an MCP server and executes tools."""
    async def mcp_tool_executor(**kwargs):
        # Connect to MCP server
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=StdioServerParameters(
                command=command,
                args=args or [],
                env=env or {},
            )
        )
        
        try:
            # Try all tools until one succeeds
            return await try_tools_sequentially(tools, kwargs, exit_stack)
        finally:
            # cleanup
            await exit_stack.aclose()
    
    return mcp_tool_executor

# Step 5: Create YouTube search function
search_youtube = create_mcp_tool_executor(
    command="mcp-youtube-search",
    args=[],
    env={"SERP_API_KEY": os.getenv("SERP_API_KEY")}
)

# Step 6: Add documentation for the LLM
search_youtube.__name__ = "search_youtube"
search_youtube.__doc__ = """
Search for YouTube videos based on a search query.
    
Args:
    search_query: The search terms to look for on YouTube (e.g., 'Google Cloud Next 25')
    max_results: Optional. Maximum number of results to return (default: 10)

Returns:
    List of YouTube videos with details including title, channel, link, published date, 
    duration, views, thumbnail URL, and description.

# Step 7: Create the agent with instructions for formatting results
agent = LlmAgent(
    name="youtube_assistant",
    model=LiteLlm(model="ollama/gemma3:12b"),
    instruction="""You are a helpful YouTube video search assistant.
Your goal is to use the search_youtube tool and present the results clearly.

1.  When asked to find videos, call the search_youtube tool.
2.  The tool will return a JSON object. Find the list of videos in the 'results' field of this JSON.
3.  For each video in the list, create a bullet point (*).
4.  Format each bullet point like this: **Title** (Link) by Channel: Description. (Published Date, Views, Duration)
    - Use the 'title', 'link', 'channel', 'description', 'published_date', 'views', and 'duration' fields from the JSON for each video.
    - Make the title bold.
    - Put the link in parentheses right after the title.
5.  Your final response should ONLY be the formatted bullet list of videos. Do not include the raw JSON.
6.  If the 'results' list in the JSON is empty, simply respond: "I couldn't find any videos for that search."
""",
    tools=[search_youtube],
)

