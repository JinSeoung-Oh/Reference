### From https://medium.com/@astropomeai/ai-agent-github-issue-resolver-an-ai-agent-for-automatic-code-generation-through-llm-and-github-b422981fcb78

from langchain_core.tools import BaseTool
import request
from langchain_openai import ChatOpenAI, OpenAIEmbedding
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langchain_core.messages import (
    SystemMessage
    HumanMessage,
    ToolMessage,
)

class GitHubIssueRetrieverInput(GitHubRepoInfo):
    pass

class GitHubIssueRetrieverTool(BaseTool):
    name: str = "github_issue_retriever"
    description: str = "Retrieves open issues from the specified GitHub repository."
    args_schema: Type[BaseModel] = GitHubIssueRetrieverInput

    def _run(
        self, tool_input: Optional[Dict[str, Any]] = None, run_manager: Optional = None, **kwargs
    ) -> str:
        if tool_input is None:
            tool_input = kwargs
        owner = tool_input.get('owner', OWNER)
        repo = tool_input.get('repo', REPO)
        access_token = tool_input.get('access_token', ACCESS_TOKEN)
        headers = {"Authorization": f"token {access_token}"}
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            issues = response.json()
            return json.dumps(issues)
        else:
            return f"Failed to retrieve issues: {response.status_code} {response.text}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("github_issue_retriever only supports sync")

  class GitHubBranchCreatorInput(GitHubRepoInfo):
    new_branch_name: str = Field(description="Name of the new branch to create")
    source_branch_name: str = Field(default="main", description="Name of the existing branch to base the new branch on")

class GitHubBranchCreatorTool(BaseTool):
    name: str = "github_branch_creator"
    description: str = "Creates a new branch in the GitHub repository."
    args_schema: Type[BaseModel] = GitHubBranchCreatorInput

    def _run(
        self, tool_input: Optional[Dict[str, Any]] = None, run_manager: Optional = None, **kwargs
    ) -> str:
        if tool_input is None:
            tool_input = kwargs
        owner = tool_input.get('owner', OWNER)
        repo = tool_input.get('repo', REPO)
        new_branch_name = tool_input['new_branch_name']
        source_branch_name = tool_input.get('source_branch_name', 'main')
        access_token = tool_input.get('access_token', ACCESS_TOKEN)
        headers = {
            "Authorization": f"token {access_token}",
            "Content-Type": "application/json",
        }

        # Get the latest commit SHA of the source branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{source_branch_name}"
        get_response = requests.get(url, headers=headers)
        if get_response.status_code == 200:
            source_branch_info = get_response.json()
            sha = source_branch_info['object']['sha']
        else:
            return f"Failed to get source branch SHA: {get_response.status_code} {get_response.text}"

        # Create the new branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
        data = {
            "ref": f"refs/heads/{new_branch_name}",
            "sha": sha
        }
        post_response = requests.post(url, headers=headers, json=data)
        if post_response.status_code == 201:
            return f"Branch '{new_branch_name}' created successfully."
        else:
            return f"Failed to create branch: {post_response.status_code} {post_response.text}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("github_branch_creator only supports sync")

  class GitHubFileUpdaterInput(GitHubRepoInfo):
    file_path: str = Field(description="Path to the file in the repository")
    content: str = Field(description="New content of the file")
    commit_message: str = Field(description="Commit message")
    branch: str = Field(description="Name of the branch to update")

class GitHubFileUpdaterTool(BaseTool):
    name: str = "github_file_updater"
    description: str = "Updates or creates a file in the GitHub repository with new content."
    args_schema: Type[BaseModel] = GitHubFileUpdaterInput

    def _run(
        self, tool_input: Optional[Dict[str, Any]] = None, run_manager: Optional = None, **kwargs
    ) -> str:
        if tool_input is None:
            tool_input = kwargs
        owner = tool_input.get('owner', OWNER)
        repo = tool_input.get('repo', REPO)
        file_path = tool_input['file_path']
        content = tool_input['content']
        commit_message = tool_input['commit_message']
        access_token = tool_input.get('access_token', ACCESS_TOKEN)
        branch = tool_input.get('branch', 'main')

        if branch == 'main':
            return "Error: Direct modifications to main branch are not allowed. Please create a new branch first."

        headers = {
            "Authorization": f"token {access_token}",
            "Content-Type": "application/json",
        }

        # Get the file SHA (specify branch)
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
        get_response = requests.get(url, headers=headers)
        if get_response.status_code == 200:
            file_info = get_response.json()
            sha = file_info['sha']
        elif get_response.status_code == 404:
            sha = None  # New file
        else:
            return f"Failed to get file SHA: {get_response.status_code} {get_response.text}"

        new_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        data = {
            "message": commit_message,
            "content": new_content,
            "branch": branch,
        }
        if sha:
            data["sha"] = sha  # Update existing file

        # Create or update the file
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        put_response = requests.put(url, headers=headers, json=data)
        if put_response.status_code in [200, 201]:
            return "File updated successfully."
        else:
            return f"Failed to update file: {put_response.status_code} {put_response.text}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("github_file_updater only supports sync")


# LLM Configuration
llm = ChatOpenAI(model_name="gpt-4o")

# Create a list of tools
tools = [
    GitHubIssueRetrieverTool(),
    GitHubFileRetrieverTool(),
    GitHubFileUpdaterTool(),
    GitHubIssueCloserTool(),
    GitHubBranchCreatorTool(),
    GitHubPullRequestCreatorTool(),
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define system prompt
system_prompt = """
You are an agent that manages a GitHub repository. Please execute tasks following the basic flow below:

1. Read open issues.
2. Create a new working branch.
3. Create or update code in the new branch.
4. Commit and push.
5. Create a pull request.

Notes:
- Never make direct changes to the main branch.
- Always create a new branch before starting work.
- Always specify the branch when updating files.
"""

# Define nodes
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Tool node
tool_node = ToolNode(tools=tools)

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


