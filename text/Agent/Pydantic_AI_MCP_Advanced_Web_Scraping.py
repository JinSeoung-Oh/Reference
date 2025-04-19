### From https://medium.com/data-science-collective/pydantic-ai-mcp-advanced-web-scraping-the-key-to-powerful-agentic-ai-e1aced88a831

#app.py
import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

deepseek_chat_model = OpenAIModel( #define the base as open AI
    'deepseek-chat',
    base_url='https://api.deepseek.com',
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# Define the MCP Servers
exa_server = MCPServerStdio(
    'python',
    ['exa_search.py']
)

python_tools_server = MCPServerStdio(
    'python',
    ['python_tools.py']
)

# Define the Agent with both MCP servers
agent = Agent(
    deepseek_chat_model, 
    mcp_servers=[exa_server, python_tools_server],
    retries=3
)

# Main async function
async def main():
    async with agent.run_mcp_servers():
        result = await agent.run("""
        I need to analyze some climate data. First, search for recent climate change statistics.
        Then, create a bar chart showing the increase in global temperature over the last decade.
        Use Python for the data visualization.
        """)
        print(result)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
-------------------------------------------------------------------------------------------------
#exa_search.py
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from exa_py import Exa

load_dotenv(override=True)

# Initialize FastMCP
mcp = FastMCP(
    name="websearch", 
    version="1.0.0",
    description="Web search capability using Exa API"
)

# Initialize the Exa client
exa_api_key = os.getenv("EXA_API_KEY", "")
exa = Exa(api_key=exa_api_key)

# Default search configuration
websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

@mcp.tool()
async def search_web(query: str, num_results: int = None) -> str:
    """Search the web using Exa API and return results as markdown formatted text."""
    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }
        
        search_results = exa.search_and_contents(
            query, 
            summary={"query": "Main points and key takeaways"},
            **search_args
        )
        
        return format_search_results(search_results)
    except Exception as e:
        return f"An error occurred while searching with Exa: {e}"

def format_search_results(search_results):
    if not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"
    for idx, result in enumerate(search_results.results, 1):
        title = result.title if hasattr(result, 'title') and result.title else "No title"
        url = result.url
        published_date = f" (Published: {result.published_date})" if hasattr(result, 'published_date') and result.published_date else ""
        
        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"
        
        if hasattr(result, 'summary') and result.summary:
            markdown_results += f"> **Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"
    
    return markdown_results

if __name__ == "__main__":
    mcp.run()

-----------------------------------------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP
import io
import base64
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback

mcp = FastMCP("python_tools")

class PythonREPL:
    def run(self, code):
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        try:
            exec(code, globals())
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {str(e)}\n{traceback.format_exc()}"

repl = PythonREPL()

@mcp.tool()
async def python_repl(code: str) -> str:
    """Execute Python code."""
    return repl.run(code)

@mcp.tool()
async def data_visualization(code: str) -> str:
    """Execute Python code. Use matplotlib for visualization."""
    try:
        repl.run(code)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()  # Close the figure to free memory
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"Error creating chart: {str(e)}"

if __name__ == "__main__":
    mcp.run()
