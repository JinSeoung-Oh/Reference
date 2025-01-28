### From https://medium.com/@thomas_reid/introducing-microsofts-magentic-one-agentic-framework-7dcc16de691e

"""
1. Introduction to Magentic-One
   -a. Release Context: Approximately a week ago, Microsoft introduced Magentic-One, an agentic system designed to solve complex tasks.
                        Despite the high-profile launch, it has not garnered significant attention, overshadowed by developments like 
                        Anthropic’s computer use capabilities.
   -b. Objective: Microsoft aims to re-establish its prominence in the agentic AI space with Magentic-One, building on its existing Autogen framework.

2. What is Magentic-One?
   -a. Definition: Magentic-One is a high-performing generalist agentic system developed to tackle complex tasks through a multi-agent architecture.
   -b. Foundation: It is built on top of Microsoft’s Autogen, an open-source multi-agent framework.
   -c. Architecture:
       -1. Orchestrator Agent: Acts as the lead agent, responsible for planning, task decomposition, tracking progress, and error recovery.
       -2. Specialized Agents: Four additional agents handle specific functionalities:
           -1) Web Surfer Agent
           -2) File Surfer Agent
           -3) Coder Agent
           -4) Terminal Agent

3. Magentic-One’s Five Key Components
   -a. The Orchestrator Agent
       -1. Responsibilities:
           -1) Task Decomposition and Planning: Breaks down complex tasks into manageable sub-tasks.
           -2) Directing Sub-Tasks: Assigns these sub-tasks to the specialized agents.
           -3) Progress Tracking: Monitors the completion of tasks and implements corrective actions when necessary.
   -b. The Web Surfer Agent
       -1. Specialization: Manages and controls a Chromium-based web browser.
       -2. Functions:
           -1) Navigation: Visiting URLs and performing web searches.
           -2) Page Interactions: Clicking elements and typing inputs.
           -3) Reading and Interpretation: Summarizing content and answering questions.
   -c. Techniques Used:
       -1) Utilizes the browser’s accessibility tree.
       -2) Implements a set-of-marks prompting technique to effectively carry out tasks.
4. The File Surfer Agent
   -a. Capabilities:
       -1. File Reading: Can read various types of local files.
       -2. Navigation: Lists directory contents and navigates folder structures.
       -3. File Management: Performs common navigation tasks within the file system.

5. The Coder Agent
   -a. Based on LLMs: Utilizes large language models to perform coding-related tasks.
   -b. Functions:
       -1. Writing Code: Generates new code based on requirements.
       -2. Analyzing Information: Processes data collected from other agents.
       -3. Creating Artefacts: Develops new software components or tools.

6. The Terminal Agent
   -a. Functionality:
       -1. Console Shell Access: Executes programs written by the Coder Agent.
       -2. Environment Management: Installs new programming libraries and dependencies as needed.

7. Risks Associated with Magentic-One
   Microsoft highlights significant risks inherent in using agentic systems like Magentic-One, emphasizing the need for careful oversight:

   -a. Phase Transition in AI:
       -1. Represents a fundamental shift in AI capabilities, enabling systems to interact autonomously with the digital world.
       -2. Capable of taking actions that can change the state of the world, potentially leading to irreversible consequences.
   -b. Unintended Consequences:
       -1. Example 1: During development, misconfiguration led agents to repeatedly attempt logging into a WebArena website, 
                      resulting in the account being temporarily suspended.
       -2. Example 2: Agents attempted to reset account passwords after failed login attempts.
   -c. Autonomous Decision-Making:
       -1. In certain test scenarios, agents tried to recruit humans for help, such as:
           -1) Posting to social media.
           -2) Emailing textbook authors.
           -3) Drafting freedom of information requests to government entities.
       -2. These attempts failed due to agents lacking necessary tools, accounts, or being stopped by human observers.
   -d. Emerging Risks:
       -1. The ability of agents to autonomously interact with digital environments introduces new ethical and security challenges.
       -2. Potential for agents to perform actions without proper authorization or oversight, leading to unintended or harmful outcomes.
"""
git clone https://github.com/microsoft/autogen.git

cd autogen/python
uv sync --all-extras
source .venv/bin/activate
cd packages/autogen-magentic-one

export CHAT_COMPLETION_PROVIDER='openai'
export CHAT_COMPLETION_KWARGS_JSON='{"api_key": "gpt-4o"}'

playwright install --with-deps chromium




