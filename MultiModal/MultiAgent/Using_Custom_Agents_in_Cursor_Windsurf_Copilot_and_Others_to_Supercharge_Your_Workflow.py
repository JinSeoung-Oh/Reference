### From https://ai.gopubby.com/using-custom-agents-in-cursor-windsurf-copilot-and-others-to-supercharge-your-workflow-f936b630c5e5
### https://github.com/BrainBlend-AI/atomic-agents

uv init
uv add mcp[cli] atomic-agents openai instructor aiohttp beautifulsoup4 markdownify readability-lxml requests pydantic lxml[html_clean] python-dotenv

### Creating a Basic MCP Server 
## server.py

# agentic_research_mcp\server.py

import logging
from mcp.server.fastmcp import FastMCP

# Set the logging level to WARNING
logging.basicConfig(level=logging.WARNING)


def main():
    # Create a new MCP server with the identifier "tutorial"
    mcp = FastMCP("tutorial")

    # Register a simple tool that returns a greeting
    @mcp.tool(name="hello", description="Returns a simple greeting message.")
    async def hello_tool(args: dict) -> str:
        return "Hello, World! This is your basic MCP server."

    @mcp.tool(
        name="string_length", description="Calculate the length of a given string."
    )
    async def string_length_tool(args: dict) -> dict:
        input_string = args.get("text", "")
        return {"text": input_string, "length": len(input_string)}

    @mcp.tool(name="reverse_strings", description="Reverse each string in the list.")
    async def reverse_strings_tool(args: dict) -> dict:
        strings = args.get("strings", [])
        reversed_strings = [
            s[::-1] for s in strings
        ]
        return {"original": strings, "reversed": reversed_strings}

    # Start the server
    mcp.run()


if __name__ == "__main__":
    main()

-----------------------------------------------------------------------------------------------
### test_client.py
# test_client.py

import asyncio
import sys
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async def main():
    # Connect to the server using the current Python interpreter
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_research_mcp.server"],
    )

    print("\n📱 Starting MCP client...\n")

    # Connect to the server and create a session
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session and get available tools
            await session.initialize()
            tools = await session.list_tools()
            print("🔧 Available tools:", ", ".join(tool.name for tool in tools.tools))

            # Call the hello tool
            response = await session.call_tool(
                name="hello",
                arguments={"args": {}},
            )
            print("\n💬 Server response (hello):", response.content[0].text)

            # Test string length tool
            test_string = "Hello, MCP!"
            response = await session.call_tool(
                name="string_length",
                arguments={"args": {"text": test_string}},
            )
            result = response.content[0].text
            print(f"\n📏 String length of '{test_string}':", result)

            # Test reverse strings tool
            test_strings = ["apple", "banana", "cherry"]
            response = await session.call_tool(
                name="reverse_strings",
                arguments={"args": {"strings": test_strings}},
            )
            result = response.content[0].text
            print(f"\n🔄 Reversing list {test_strings}:", result, "\n")

if __name__ == "__main__":
    asyncio.run(main())

-----------------------------------------------------------------------------
### uv run atomic
import re
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from pydantic import Field, HttpUrl
from readability import Document

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig


class WebpageScraperToolInputSchema(BaseIOSchema):
    """Schema for webpage scraper input."""

    url: HttpUrl = Field(..., description="URL of the webpage to scrape.")


class WebpageMetadata(BaseIOSchema):
    """Schema for webpage metadata."""

    title: str = Field(..., description="The title of the webpage.")
    domain: str = Field(..., description="Domain name of the website.")
    description: Optional[str] = Field(
        None, description="Meta description of the webpage."
    )


class WebpageScraperToolOutputSchema(BaseIOSchema):
    """Schema for webpage scraper output."""

    content: str = Field(..., description="The scraped content in markdown format.")
    metadata: WebpageMetadata = Field(
        ..., description="Metadata about the scraped webpage."
    )


class WebpageScraperToolConfig(BaseToolConfig):
    """Configuration for the webpage scraper tool."""

    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        description="User agent string to use for requests.",
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for HTTP requests.",
    )


class WebpageScraperTool(BaseTool):
    """Tool for scraping webpage content."""

    input_schema = WebpageScraperToolInputSchema
    output_schema = WebpageScraperToolOutputSchema

    def __init__(self, config: WebpageScraperToolConfig = WebpageScraperToolConfig()):
        super().__init__(config)
        self.config = config

    def _fetch_webpage(self, url: str) -> str:
        """Fetches webpage content."""
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = requests.get(url, headers=headers, timeout=self.config.timeout)
        return response.text

    def _extract_metadata(
        self, soup: BeautifulSoup, doc: Document, url: str
    ) -> WebpageMetadata:
        """Extracts metadata from the webpage."""
        domain = urlparse(url).netloc
        description = None

        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag:
            description = description_tag.get("content")

        return WebpageMetadata(
            title=doc.title(),
            domain=domain,
            description=description,
        )

    def _clean_markdown(self, markdown: str) -> str:
        """Cleans up markdown content."""
        markdown = re.sub(r"\n\s*\n\s*\n", "\n\n", markdown)
        markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
        markdown = markdown.strip() + "\n"
        return markdown

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extracts main content from webpage."""
        for element in soup.find_all(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        content_candidates = [
            soup.find("main"),
            soup.find(id=re.compile(r"content|main", re.I)),
            soup.find(class_=re.compile(r"content|main", re.I)),
            soup.find("article"),
        ]

        main_content = next((c for c in content_candidates if c), None)
        if not main_content:
            main_content = soup.find("body")

        return str(main_content) if main_content else str(soup)

    def run(
        self, params: WebpageScraperToolInputSchema
    ) -> WebpageScraperToolOutputSchema:
        """Runs the webpage scraper tool."""
        html_content = self._fetch_webpage(str(params.url))
        soup = BeautifulSoup(html_content, "html.parser")
        doc = Document(html_content)

        main_content = self._extract_main_content(soup)
        markdown_content = markdownify(
            main_content,
            strip=["script", "style"],
            heading_style="ATX",
            bullets="-",
        )
        markdown_content = self._clean_markdown(markdown_content)
        metadata = self._extract_metadata(soup, doc, str(params.url))

        return WebpageScraperToolOutputSchema(
            content=markdown_content,
            metadata=metadata,
        )

----------------------------------------------------------------------------------------------
"""Configuration settings for the MCP server."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChatConfig:
    """Configuration settings for chat models."""

    # OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Default model to use
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

-----------------------------------------------------------------------------------------------
"""Query generation agent for the Deep Research MCP server."""

import instructor
import openai
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from ..tools.tavily_search import TavilySearchToolInputSchema
from ..config import ChatConfig


class QueryAgentInputSchema(BaseIOSchema):
    """Input schema for the QueryAgent."""

    instruction: str = Field(
        ...,
        description="A detailed instruction or request to generate search engine queries for.",
    )
    num_queries: int = Field(
        ..., description="The number of search queries to generate."
    )


def create_query_agent() -> BaseAgent:
    """Creates and configures a new query generation agent."""
    return BaseAgent(
        BaseAgentConfig(
            client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
            model=ChatConfig.model,
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    "You are an expert search engine query generator with a deep understanding of which queries will maximize relevant results."
                ],
                steps=[
                    "Analyze the given instruction to identify key concepts",
                    "For each aspect, craft a search query using appropriate operators",
                    "Ensure queries cover different angles (technical, practical, etc.)",
                ],
                output_instructions=[
                    "Return exactly the requested number of queries",
                    "Format each query like a search engine query, not a question",
                    "Each query should be concise and use relevant keywords",
                ],
            ),
            input_schema=QueryAgentInputSchema,
            output_schema=TavilySearchToolInputSchema,
        )
    )

--------------------------------------------------------------------------------------------
"""Question answering agent for the Deep Research MCP server."""

import instructor
import openai
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from agentic_research_mcp.tools.webpage_scraper import WebpageScraperToolOutputSchema

from ..config import ChatConfig



class QuestionAnsweringAgentInputSchema(BaseIOSchema):
    """Input schema for the QuestionAnsweringAgent."""

    question: str = Field(..., description="The question to answer.")
    context: list[WebpageScraperToolOutputSchema] = Field(
        ...,
        description="List of scraped webpages used to generate the answer.",
    )


class QuestionAnsweringAgentOutputSchema(BaseIOSchema):
    """Output schema for the QuestionAnsweringAgent."""

    answer: str = Field(..., description="The answer to the question.")


def create_qa_agent() -> BaseAgent:
    """Creates and configures a new question answering agent."""
    return BaseAgent(
        BaseAgentConfig(
            client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
            model=ChatConfig.model,
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    "You are an expert research assistant focused on providing accurate, well-sourced information.",
                    "Your answers should be based on the provided web content and include relevant source citations.",
                ],
                steps=[
                    "Analyze the question and identify key information needs",
                    "Review all provided web content thoroughly",
                    "Synthesize information from multiple sources",
                    "Formulate a clear, comprehensive answer",
                ],
                output_instructions=[
                    "Answer should be detailed but concise",
                    "Include specific facts and data from sources",
                    "If sources conflict, acknowledge the discrepancy",
                    "If information is insufficient, acknowledge limitations",
                ],
            ),
            input_schema=QuestionAnsweringAgentInputSchema,
            output_schema=QuestionAnsweringAgentOutputSchema,
        )
    )

-------------------------------------------------------------------------------------------------
# agentic_research_mcp\server.py

import logging
import os
import json
import traceback
from mcp.server.fastmcp import FastMCP

# Import required components for the web search pipeline
from agentic_research_mcp.agents.query_agent import create_query_agent, QueryAgentInputSchema
from agentic_research_mcp.agents.qa_agent import create_qa_agent, QuestionAnsweringAgentInputSchema
from agentic_research_mcp.tools.tavily_search import (
    TavilySearchTool,
    TavilySearchToolConfig,
    TavilySearchToolInputSchema
)
from agentic_research_mcp.tools.webpage_scraper import (
    WebpageScraperTool,
    WebpageScraperToolInputSchema
)

# Set up logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Create a new MCP server with the identifier "tutorial"
    mcp = FastMCP("research_pipeline")
    logger.info("Starting research pipeline server...")

    try:
        # Initialize the components
        logger.info("Initializing query agent...")
        query_agent = create_query_agent()

        logger.info("Initializing QA agent...")
        qa_agent = create_qa_agent()

        # Initialize Tavily search tool with API key from environment
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_api_key:
            logger.error("TAVILY_API_KEY environment variable is not set. Web search will not work properly.")
            raise ValueError("TAVILY_API_KEY environment variable must be set")

        logger.info("Initializing Tavily search tool...")
        tavily_tool = TavilySearchTool(
            config=TavilySearchToolConfig(
                api_key=tavily_api_key,
                max_results=5,  # Limiting to top 5 results per query
                include_answer=True
            )
        )

        logger.info("Initializing web scraper tool...")
        scraper_tool = WebpageScraperTool()

        @mcp.tool(
            name="web_search_pipeline",
            description="Performs a web search pipeline: generates queries, searches web, sorts results, scrapes pages, and answers questions."
        )
        async def web_search_pipeline(args: dict) -> str:
            # Extract the instruction and question
            instruction = args.get("instruction", "")
            question = args.get("question", instruction)  # Use instruction as question if not provided
            num_queries = args.get("num_queries", 3)

            logger.info(f"Starting web search pipeline for question: {question}")

            try:
                # Step 1: Generate search queries using query agent
                logger.info(f"Step 1: Generating {num_queries} search queries...")
                query_input = QueryAgentInputSchema(instruction=instruction, num_queries=num_queries)
                query_result = query_agent.run(query_input)
                queries = query_result.queries
                logger.info(f"Generated queries: {queries}")

                # Step 2: Perform web search using Tavily
                logger.info(f"Step 2: Performing web search with {len(queries)} queries...")
                search_input = TavilySearchToolInputSchema(queries=queries)
                search_results = tavily_tool.run(search_input)
                logger.info(f"Received {len(search_results.results)} search results")

                # Step 3: Sort results by score in descending order
                logger.info("Step 3: Sorting results by score...")
                sorted_results = sorted(search_results.results, key=lambda x: x.score, reverse=True)
                logger.info(f"Top result score: {sorted_results[0].score if sorted_results else 'No results'}")

                # Step 4: Take top results and scrape their content
                top_results = sorted_results[:5]  # Limit to top 5 results
                scraped_pages = []
                logger.info(f"Step 4: Scraping content from top {len(top_results)} results...")

                for i, result in enumerate(top_results):
                    try:
                        logger.info(f"Scraping {i+1}/{len(top_results)}: {result.url}")
                        scrape_input = WebpageScraperToolInputSchema(url=result.url)
                        scrape_result = scraper_tool.run(scrape_input)
                        scraped_pages.append(scrape_result)
                        logger.info(f"Successfully scraped {result.url}")
                    except Exception as e:
                        logger.error(f"Error scraping {result.url}: {str(e)}")
                        logger.error(traceback.format_exc())

                # Step 5: Generate answer using QA agent
                logger.info("Step 5: Generating answer using QA agent...")
                qa_input = QuestionAnsweringAgentInputSchema(question=question, context=scraped_pages)
                qa_result = qa_agent.run(qa_input)
                logger.info("Answer generated successfully")

                # Return comprehensive result as JSON string
                result = {
                    "question": question,
                    "queries_generated": queries,
                    "search_results": [
                        {
                            "title": result.title,
                            "url": result.url,
                            "score": float(result.score)  # Ensure score is serializable
                        } for result in sorted_results[:10]  # Include top 10 search results in response
                    ],
                    "answer": qa_result.markdown_output if hasattr(qa_result, 'markdown_output') else qa_result.answer,
                    "references": qa_result.references if hasattr(qa_result, 'references') else [],
                    "followup_questions": qa_result.followup_questions if hasattr(qa_result, 'followup_questions') else []
                }
                logger.info("Pipeline completed successfully")
                return json.dumps(result)  # Return JSON string

            except Exception as e:
                logger.error(f"Error in pipeline execution: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = {
                    "error": str(e),
                    "question": question,
                    "stage": "pipeline execution",
                    "traceback": traceback.format_exc()
                }
                return json.dumps(error_response)  # Return error as JSON string

        logger.info("All components initialized successfully. Starting server...")
        # Start the server
        mcp.run()

    except Exception as e:
        logger.error(f"Error during server initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

--------------------------------------------------------------------------------------------
# test_client.py

import asyncio
import sys
import json
import logging
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    # Connect to the server using the current Python interpreter
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_research_mcp.server"],
    )

    logger.info("\n📱 Starting MCP client...\n")

    try:
        # Connect to the server and create a session
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the session and get available tools
                await session.initialize()
                tools = await session.list_tools()
                logger.info("🔧 Available tools: %s", ", ".join(tool.name for tool in tools.tools))

                # Call the hello tool
                logger.info("Testing hello tool...")
                response = await session.call_tool(
                    name="hello",
                    arguments={"args": {}},
                )
                logger.info("\n💬 Server response (hello): %s", response.content[0].text)

                # Test the web search pipeline
                logger.info("\n🔍 Testing web search pipeline...")
                search_question = "What are the latest advancements in quantum computing?"

                try:
                    logger.info("\n❓ Question: %s", search_question)
                    logger.info("Calling web_search_pipeline tool...")

                    response = await session.call_tool(
                        name="web_search_pipeline",
                        arguments={
                            "args": {
                                "instruction": search_question,
                                "num_queries": 3
                            }
                        },
                    )

                    logger.info("Received response from web_search_pipeline")
                    response_text = response.content[0].text
                    logger.info("Raw response: %s", response_text)

                    # Parse and display the JSON response
                    result = json.loads(response_text)

                    if "error" in result:
                        logger.error("Pipeline error: %s", result["error"])
                        if "traceback" in result:
                            logger.error("Traceback: %s", result["traceback"])
                        return

                    # Print generated queries
                    print("\n🔎 Generated Search Queries:")
                    for i, query in enumerate(result.get("queries_generated", [])):
                        print(f"  {i+1}. {query}")

                    # Print top search results
                    print("\n📊 Top Search Results:")
                    for i, result_item in enumerate(result.get("search_results", [])[:5]):  # Show top 5
                        print(f"  {i+1}. {result_item['title']} ({result_item['score']:.4f})")
                        print(f"     URL: {result_item['url']}")

                    # Print the generated answer
                    print("\n📝 Generated Answer:")
                    print(result.get("answer", "No answer generated"))

                    # Print references if available
                    if result.get("references"):
                        print("\n📚 References:")
                        for i, ref in enumerate(result["references"]):
                            print(f"  {i+1}. {ref}")

                    # Print follow-up questions if available
                    if result.get("followup_questions"):
                        print("\n❓ Suggested Follow-up Questions:")
                        for i, question in enumerate(result["followup_questions"]):
                            print(f"  {i+1}. {question}")

                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON response: %s", str(e))
                    logger.error("Raw response content: %s", response.content[0].text if response and response.content else "No content")
                except Exception as e:
                    logger.error("Error in web search pipeline: %s", str(e))
                    import traceback
                    logger.error("Traceback: %s", traceback.format_exc())

    except Exception as e:
        logger.error("Error connecting to server: %s", str(e))
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())

----------------------------------------------------------------------------------------

[project]
name = "agentic-research-mcp"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "aiohttp>=3.11.12",
    "atomic-agents>=1.0.21",
    "beautifulsoup4>=4.13.3",
    "instructor>=1.7.2",
    "lxml[html-clean]>=5.3.1",
    "markdownify>=0.14.1",
    "mcp[cli]>=1.3.0",
    "openai>=1.63.2",
    "pydantic>=2.10.6",
    "python-dotenv>=1.0.1",
    "readability-lxml>=0.8.1",
    "requests>=2.32.3",
]

[project.scripts]
agentic-research = "agentic_research_mcp.server:main"

---------------------------------------------------------------------------------------



