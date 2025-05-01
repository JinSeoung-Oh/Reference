### From https://medium.com/data-and-beyond/browser-use-explained-the-open-source-ai-agent-that-clicks-reads-and-automates-the-web-d4689f3ef012

"""
Browser Use is a Python library that empowers large-language-model–driven agents (e.g. GPT-4, Claude) to operate a real web browser 
via natural-language instructions—no hand-coded selectors or Playwright/Selenium scripts required. 
Under the hood it leverages Microsoft’s Playwright for reliability,
but all decision-making (“what to click,” “where to scroll,” “when to fill a form”) is handled by an LLM that reasons 
over both the DOM structure and visual screenshots.

1. The Problem It Solves
   -a. Selector and Timing Fragility
       -1. Traditional automation forces you to hunt down CSS/XPath selectors that often break when page layouts or JavaScript change.
       -2. You spend hours tweaking waits or retries to handle dynamic loading.
       -3. Browser Use sidesteps this by having the LLM “see” and interpret page elements the way a person would—if a button moves, 
           it can still find and click it.
   -b. Boilerplate Overhead
       -1. Logging in, navigating multi-step flows, filling out forms—all require dozens of lines of monotonous code.
       -2. With Browser Use, you simply describe your goal (“book me the cheapest nonstop flight”) and the agent composes the entire script on the fly.
   -c. Lack of High-Level Abstraction
       -1. Traditional tools excel at “click here, scroll there” recipes, but real tasks are goal-oriented 
           (“find top 5 remote QA jobs and list their URLs”), not just hard-coded sequences.
       -2. Browser Use frames web automation as a reasoning task—your instructions remain high-level, 
           and the agent breaks them down into page interactions.

2. Key Benefits
   -a. Natural-Language Tasking
       -1. Describe what you want, not how to accomplish it.
   -b. Adaptive Robustness
       -1. If a site’s structure shifts or a pop-up appears, the LLM can adjust on the fly.
   -c. Visual + DOM Context
       -1. Combines raw HTML access with rendered screenshots so the agent understands both code and appearance.
   -d. Minimal Scripting Effort
       -1. Zero boilerplate for authentication flows, pagination, or dynamic content handling.
"""

uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

!pip install "browser-use[memory]"
!playwright install


from browser_use import Agent
from langchain_openai import ChatOpenAI
import asyncio
from browser_use import BrowserConfig
from browser_use import Browser

OPENAI_API_KEY="your-openai-key"
## ANTHROPIC_API_KEY="your-claude-key"

browser_config = BrowserConfig(headless=False)
browser = Browser(config=BrowserConfig(
    browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
))


agent = Agent(
    task="Find the latest news about generative AI and summarize it in 3 sentences.",
    llm=ChatOpenAI(model="gpt-4o")
)
async def main():
    result = await agent.run()
    print(result)
asyncio.run(main())

-------------------------------------------------------------------------------------

from browser_use import Agent, Controller, ActionResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import asyncio

# Step 1: Define the schema
class Post(BaseModel):
    title: str
    url: str
    comments: int
class TopPosts(BaseModel):
    posts: list[Post]
# Step 2: Add a custom action (just for fun here)
controller = Controller()
@controller.action("Save to file")
def save_to_file(filename: str, content: str) -> str:
    with open(filename, "w") as f:
        f.write(content)
    return ActionResult(extracted_content=f"Saved to {filename}")
# Step 3: Run the agent
agent = Agent(
    task="Go to example.com/forum and extract the top 5 posts with their title, URL, and number of comments.",
    llm=ChatOpenAI(model="gpt-4o"),
    controller=controller,
    output_model=TopPosts
)
async def main():
    result = await agent.run()
    print(result.json(indent=2))
asyncio.run(main())

-------------------------------------------------------------------------------------

from browser_use import Agent, Browser, BrowserConfig, BrowserContext
from browser_use.browser.context import BrowserContextConfig
from langchain_openai import ChatOpenAI
import asyncio

# Configure headed browser for visibility
browser_config = BrowserConfig(headless=False, disable_security=True)
browser = Browser(config=browser_config)
# Define layout, locale
context_config = BrowserContextConfig(
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    wait_for_network_idle_page_load_time=3.0
)
context = BrowserContext(browser=browser, config=context_config)
# Define the task
task = "Find the cheapest nonstop flight from Dubai to Cochin (COK) for tomorrow in economy for one adult."
agent = Agent(task=task, llm=ChatOpenAI(model="gpt-4o"), browser_context=context)
async def main():
    result = await agent.run()
    print("Cheapest Flight Found:", result)
asyncio.run(main())

