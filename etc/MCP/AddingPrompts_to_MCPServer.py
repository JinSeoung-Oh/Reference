### From https://blog.stackademic.com/adding-prompts-to-mcp-server-64d058c4e758

### Create Prompts
from mcp.server.fastmcp.prompts import Prompt
from mcp.server.fastmcp.prompts.base import PromptArgument

example_prompt = Prompt(
    name="PROMPT_NAME",
    description="PROMPT_DESCRIPTION",
    arguments=[
        PromptArgument(
            name="ARGUMENT_NAME",
            description="ARGUMENT_DESCRIPTION",
            required="OPTIONAL_OR_REQUIRED"
        )
    ],
    fn="FUNCTION_FOR_BUILDING_THE_PROMPT"
)

--------------------------------------------------------------------------------------------
### prompts.py

from mcp.server.fastmcp.prompts import Prompt
from mcp.server.fastmcp.prompts.base import PromptArgument
from prompt_builder import PromptBuilder

text_refinement_prompt = Prompt(
    name="refine-the-text",
    description="Refine the given text",
    arguments=[],
    fn=PromptBuilder.text_refinement_prompt_fn,
)

bugfix_prompt = Prompt(
    name="fix-the-issue",
    description="Fix the issue in the give code snippet",
    arguments=[
        PromptArgument(
            name="framework",
            description="The framework used in the code snippet",
            required=True
        ),
        PromptArgument(
            name="issue",
            description="The issue in the code snippet"
        ),
    ],
    fn=PromptBuilder.bugfix_prompt_fn
)

--------------------------------------------------------------------------------------------
### Create Prompt Functions
## prompt_builder.py

from textwrap import dedent


class PromptBuilder:

    @staticmethod
    def text_refinement_prompt_fn():
        return dedent(
            f"""
            Rewrite the given text for grammar errors. 
            Make it more readable. 
            But don't change the meaning and writing style.
            """
        )

    @staticmethod
    def bugfix_prompt_fn(framework, issue=""):
        return dedent(
            f"""
            Act as a experienced {framework} developer. 
            Study the given code and fix the bug. 
            Bug: {issue}
            """
        )    

--------------------------------------------------------------------------------------------
## Attach To MCP Server
from mcp.server.fastmcp import FastMCP
from prompts import bugfix_prompt, text_refinement_prompt

# Initialize FastMCP server
mcp = FastMCP("Weather-Server")

# Attach prompts to the server
mcp.add_prompt(bugfix_prompt)
mcp.add_prompt(text_refinement_prompt)
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()

--------------------------------------------------------------------------------------------
## Connect To Claude Desktop
{
    "mcpServers": {
        "weather": {
            // Run python command within the virtual environment
            // Absolute path to the python exe within the virtual envirnment
            "command": "<PATH_TO_ENVIRONMENT>\\venv\\Scripts\\python",
            "args": [
                // Absolute path to server.py
                "<PATH_TO_SERVER>\\server.py" 
            ]
        }
    }
}




