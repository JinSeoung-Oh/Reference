### From https://pub.towardsai.net/building-a-multi-agent-system-with-multiple-mcp-servers-using-smolagents-95e9cedb1334

!pip install "mcp[cli]"

### Simple Markdown Memory for agent:
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import re
import sys
import signal

load_dotenv()


def exit_gracefully(signum, frame):
    print("Exiting MCP server...")
    sys.exit(0)


signal.signal(signal.SIGINT, exit_gracefully)

# Initialize the MCP server
mcp = FastMCP(name="memory_mcp",
            host="127.0.0.1",
            port=5000,
            timeout=30)


USER_AGENT = "memory/0.0.1"

# Directory in which memory markdown files will be stored.
MEMORY_DIR = r"{path_to_wanted_memory_location}/memory_files"
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)


def memory_file_path(title: str) -> str:
    # Remove any characters that aren't alphanumeric, spaces, underscores, or hyphens.
    sanitized = re.sub(r"[^\w\s\-]", "", title).strip()
    # Replace spaces with underscores for the file name.
    filename = "_".join(sanitized.split()) + ".md"
    return os.path.join(MEMORY_DIR, filename)


@mcp.tool()
async def create_memory(title: str, summary: str) -> str:
    """
    Creates a memory file (markdown) with a given title and summary.
    The file starts with a title header and is saved with a sanitized file name.
    """
    path = memory_file_path(title)
    if os.path.exists(path):
        return f"Memory '{title}' already exists. Use update_memory to add new content."
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(summary)
    return f"Memory '{title}' created successfully."


@mcp.tool()
async def get_memory_list() -> str:
    """
    Returns a string containing all memory titles found in the memory files directory, one title per line.
    """
    titles = []
    for file in os.listdir(MEMORY_DIR):
        if file.endswith(".md"):
            title = file[:-3]  # remove ".md"
            titles.append(title)
    return "\n".join(titles)


@mcp.tool()
async def update_memory(title: str, additional_summary: str) -> str:
    """
    Updates an existing memory by appending new content.
    If the memory does not exist, it creates a new memory with the provided content.
    """
    path = memory_file_path(title)
    if os.path.exists(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + additional_summary)
        return f"Memory '{title}' updated successfully."
    else:
        # If the memory doesn't exist, call create_memory.
        return await create_memory(title, additional_summary)


@mcp.tool()
async def delete_memory(title: str) -> str:
    """
    Deletes the memory file corresponding to the given title.
    """
    path = memory_file_path(title)
    if os.path.exists(path):
        os.remove(path)
        return f"Memory '{title}' deleted successfully."
    else:
        return f"Memory '{title}' does not exist."


@mcp.tool()
async def get_memory(title: str) -> str:
    """
    Retrieves the full contents of a memory file matching the given title.
    If an exact match is not found, it will search for a memory whose title contains the given string.
    """
    # Attempt to get an exact match.
    path = memory_file_path(title)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # Otherwise, attempt a partial match search.
    for file in os.listdir(MEMORY_DIR):
        if file.endswith(".md"):
            filepath = os.path.join(MEMORY_DIR, file)
            with open(filepath, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if title.lower() in first_line.lower():
                    with open(filepath, "r", encoding="utf-8") as f_full:
                        return f_full.read()
    return f"Memory matching '{title}' not found."

# ToDo list for LLM if needed
@mcp.tool()
async def todo_longterm(action: str, content: str = None) -> str:
    """
    Manages the long-term to-do list stored as a special memory.
    Actions:
      - 'get': Retrieves the current to-do list.
      - 'update': Appends a new item to the to-do list.
      - 'clear': Clears the entire to-do list.
    """
    todo_path = os.path.join(MEMORY_DIR, "todo_longterm.md")
    if action == "get":
        if os.path.exists(todo_path):
            with open(todo_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "No longterm to-do list exists."
    elif action == "update":
        if not content:
            return "No content provided for updating the to-do list."
        # Append new content as a bullet point.
        with open(todo_path, "a", encoding="utf-8") as f:
            f.write("\n- " + content)
        return "Longterm to-do list updated."
    elif action == "clear":
        with open(todo_path, "w", encoding="utf-8") as f:
            f.write("")
        return "Longterm to-do list cleared."
    else:
        return "Invalid action. Please use 'get', 'update', or 'clear'."

if __name__ == "__main__":
    print("Starting MCP server... on 127.0.0.1:5000")
    mcp.run(transport='stdio')

---------------------------------------

### PubMed Server installation and Smithery

git clone https://github.com/JackKuo666/PubMed-MCP-Server.git
cd PubMed-MCP-Server
# Then install all the dependencies:
pip install -r requirements.txt

--------------------------------------

#### Smolagents: A Simple Library for Building Agents

pip install google-generativeai
pip install smolagents[litellm]
pip install "smolagents[mcp]"

-------------------------------------

# Configuration Constants
MCP_SERVERS = {
    "pubmed": {
        "command": "python",
        "args": ["{path_to_PubMed_folder}/PubMed-MCP-Server/pubmed_server.py"],
    },
    "markdown": {
        "command": "python",
        "args": ["{path_to_custom_mcp}/mcp_memory.py"],
        "host": "127.0.0.1",
        "port": 5000,
        "timeout": 3000,
        "env": {"UV_PYTHON": "3.12", **os.environ},
    }
}

class AgentManager:
    def __init__(self):
        self.model = self._initialize_model()
        self.history: List[str] = []

    def _initialize_model(self) -> LiteLLMModel:
        """Initialize and return the LiteLLM model instance"""
        return LiteLLMModel(
            model_id="gemini/gemini-2.5-flash-preview-04-17",
            api_key=os.getenv('GEMINI_API'),
        )

    def _create_server_parameters(self, server_name: str) -> StdioServerParameters:
        """Create MCP server parameters from configuration"""
        config = MCP_SERVERS[server_name]
        return StdioServerParameters(
            command=config["command"],
            args=config["args"],
            host=config.get("host"),
            port=config.get("port"),
            timeout=config.get("timeout"),
            env=config.get("env"),
        )

    def _handle_rate_limiting(self, start_time: float, loop_count: int) -> tuple:
        """Manage API rate limiting constraints"""
        elapsed = time.time() - start_time
        if elapsed > 80:
            return time.time(), 0
        if loop_count >= 8 and elapsed < 60:
            print("Rate limit approached. Taking 20-second break...")
            time.sleep(20)
            return time.time(), 0
        return start_time, loop_count

    def run_chat_loop(self):
        """Main execution loop for user interaction"""
        start_time = time.time()
        loop_count = 0
        with (
            ToolCollection.from_mcp(
                self._create_server_parameters("markdown"), 
                trust_remote_code=True
            ) as markdown_tools,
            ToolCollection.from_mcp(
                self._create_server_parameters("pubmed"), 
                trust_remote_code=True
            ) as pubmed_tools
        ):
            # Memory Agent Configuration
            memory_agent = CodeAgent(
                tools=[*markdown_tools.tools],
                model=self.model,
                max_steps=5,
                name="memory_agent",
                description=(
                    "Memory agent that can create and update memories from markdown files. It can also retrieve memories. It can also create and update to-do lists. \
                    This agent should be called for any memory related tasks and create memory if any information is important. \
                    All the memories and todo lists are in markdown format. You should always check if there is any useful memory before planning and taking action. Save useful information in memory."
                )
            )

            # Main Agent Configuration
            main_agent = CodeAgent(
                tools=[*pubmed_tools.tools],
                model=self.model,
                managed_agents=[memory_agent],
                additional_authorized_imports=["time", "numpy", "pandas", "os"],
                description=(
                    "You are the manager agent that create a plan to finish the task then execute it in order to finish it. \
                    You should call the memory_agent for any memory related tasks like saving important information or remembering other important memories.\
                    You should always check if there is any useful memory before planning and taking action. \
                    Before searching the web with search_agent you should always check what kind of memory titles you have."
                )
            )
            while True:
                try:
                    # User input handling
                    user_input = input("User: ")
                    self.history.append(f"User: {user_input}")

                    # Generate agent response
                    prompt = "\n".join(self.history[-4:]) + "\nAgent: "
                    response = main_agent.run(prompt)

                    # Update conversation history
                    self.history.append(f"Agent: {response}")
                    print(Text("\n".join(self.history[-4:]), style="green"))

                    # Rate limiting management
                    loop_count += 1
                    start_time, loop_count = self._handle_rate_limiting(start_time, loop_count)

                except KeyboardInterrupt:
                    print("\nExiting chat session...")
                    break


if __name__ == "__main__":
    manager = AgentManager()
    manager.run_chat_loop()

