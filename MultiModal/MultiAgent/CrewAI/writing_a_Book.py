### From https://blog.gopenai.com/building-a-muti-agent-system-for-writing-a-book-crew-ai-65571624c740

!pip3 install crewai

import os
from crewai import Agent, Task, Crew, Process

#Defining the Agents

# Set environment variables for OpenAI API
os.environ["OPENAI_API_KEY"] = "Paste your key here"
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"


planning_agent = Agent(
    role="Planning Agent",
    goal="Develop the book's concept, outline, characters, and world.",
    backstory="An experienced author specializing in planning and structuring novels.",
    verbose=True
)

# Define the Writing Agent
writing_agent = Agent(
    role="Writing Agent",
    goal="Write detailed chapters based on the provided outline and character details.",
    backstory="A creative writer adept at bringing stories to life.",
    verbose=True
)

# Define the Editing Agent
editing_agent = Agent(
    role="Editing Agent",
    goal="Edit the written chapters for clarity, coherence, and grammatical accuracy.",
    backstory="A meticulous editor with an eye for detail.",
    verbose=True
)

# Define the Fact-Checking Agent
fact_checking_agent = Agent(
    role="Fact-Checking Agent",
    goal="Verify the accuracy of all factual information presented in the book.",
    backstory="A diligent researcher ensuring all facts are correct.",
    verbose=True
)

# Define the Publishing Agent
publishing_agent = Agent(
    role="Publishing Agent",
    goal="Format the manuscript and prepare it for publication.",
    backstory="An expert in publishing standards and formatting.",
    verbose=True
)

# Define the tasks for each agent

tasks = [
    Task(
        description="Develop the book's concept, outline, characters, and world.",
        expected_output="A comprehensive plan including theme, genre, outline, character profiles, and world details.",
        agent=planning_agent
    ),
    Task(
        description="Write detailed chapters based on the provided outline and character details.Each chapter should be 1000 words atleast",
        expected_output="Drafts of all chapters in the book.",
        agent=writing_agent
    ),
    Task(
        description="Edit the written chapters for clarity, coherence, and grammatical accuracy.",
        expected_output="Edited versions of all chapters.",
        agent=editing_agent
    ),
    Task(
        description="Verify the accuracy of all factual information presented in the book.",
        expected_output="A report confirming the accuracy of all facts or detailing necessary corrections.",
        agent=fact_checking_agent
    ),
    Task(
        description="Format the manuscript and prepare it for publication.",
        expected_output="A finalized manuscript ready for publication.",
        agent=publishing_agent
    )
]

# Assembling all the Agents 

book_writing_crew = Crew(
    agents=[planning_agent, writing_agent, editing_agent, fact_checking_agent, publishing_agent],
    tasks=tasks,
    process=Process.sequential,
    verbose=True
)

# Execute the workflow
if __name__ == "__main__":
    result = book_writing_crew.kickoff()
    print("Final Manuscript:", result)
