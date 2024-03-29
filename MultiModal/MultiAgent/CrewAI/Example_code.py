## From https://generativeai.pub/the-ai-symphony-testing-crewais-multi-agent-system-for-complex-apps-b377d4f5e6b8
import os
from crewai import Crew
from crewai import Agent
from crewai import Task
from textwrap import dedent
from langchain_openai import ChatOpenAI

from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
load_dotenv()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

search_tool = DuckDuckGoSearchRun()

class AssignmentHelperAgents:
  def __init__(self):
    self.OpenAIGPT41106 = ChatOpenAI(model_name="gpt-4-1106-preview ", temperature=0.3)
    self.OpenAIGPT40125 = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.3)
    self.OpenAIGPT3Turbo= ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.3)

    def IdeaGeneratorAgent(self):
        return Agent(
            role="Report Idea Generator",
            backstory=dedent("""
               You help students with their research. Lets say a student wants to write a report on famous Canadians. 
               In this case you will search and generate ideas to explore. You will not write the report, but generate ideas
            """),
            goal=dedent("""
                Use the tools you have to search and generate ideas for the user.
            """),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT40125,
            tools=[search_tool, wikipedia, wikidata]
        )

    def ReportWriterAgent(self):
        return Agent(
            role="Report Writer",
            backstory=dedent("""
                You help students with writing reports for their projects. 
                You get the ideas from the Idea Generator and then use the tools you have to write the report. 
                If feedback is give to you on your report, then you will revise your report
            """),
            goal=dedent("""
                You help write the report based on the ideas generated by the Idea Generator. You will structure the report based on direction by the user.
                A report should have these sections: Introduction, Body, Conclusion.
                You will use the Search, Wikipedia and Wikidata tools you have to collect data to write the report. If feedback is given to you then you will use the feedback to revise the report
            """),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT40125,
            tools=[search_tool, wikipedia, wikidata]
        )

    def ReportEvaluatorAgent(self):
        return Agent(
            role="Report Evaluator",
            backstory=dedent("""\
                You look through the written report and identify any opportunities for improvement
                """),
            goal=dedent("""\
                You review the written report and check it against the rubric - 
                1. Does it meets the user criteria, 2. Is it coherent. 3. Is it the appropriate length. 
                4. Does it has an intro, body and conclusion,
                 5. Is it interesting, 6. Is it accurate
                You can use the tools you have to check the accuracy of the report
                . Based on your observation you provide opportunities for improvement or a pass
            """),
            allow_delegation=True,
            llm=self.OpenAIGPT40125,
            verbose=True,
            tools=[search_tool, wikidata, wikipedia]
        )

class AssignmentWritingTasks:

    def generate_ideas(self, agent, task):
        return Task(
            description=dedent(
            f"""
            Users task is below. Use the search tools you have to generate ideas for the report. Only generate ideas.
            {task}
            """
            ),
            agent=agent,
        )

    def write_report(self, agent, task):
        return Task(
            description=dedent(
            f"""
            Use the ideas from the idea generator to now write a report for the student task below. 
            {task} 
            """
            ),
            agent=agent,
        )

    def evaluate_report(self, agent, task):
        return Task(
            description=dedent(
            f"""
            Users task is below. Review the report written and provide opportunities for improvement. Check it against your rubric
            {task} 
            """
            ),
            agent=agent,
        )
    
    def rewrite_report(self, agent, task):
        return Task(
            description=dedent(
            f"""
            Users task is below. Review the initial report and the feedback from the report critic to rewrite addressing the feedback
            {task} 
            """
            ),
            agent=agent,
        )

agents = AssignmentHelperAgents()
tasks = AssignmentWritingTasks()

## Instantiate the agents
idea_generator_agent = agents.IdeaGeneratorAgent()
writer_agent = agents.ReportWriterAgent()
evaluator_agent = agents.ReportEvaluatorAgent()

## Instantiate the tasks
generate_ideas = tasks.generate_ideas(idea_generator_agent, task_description)
write_report = tasks.write_report(writer_agent, task_description)
evaluate_report = tasks.evaluate_report(evaluator_agent, task_description)
rewrite_report = tasks.rewrite_report(writer_agent, task_description)

crew_startup = Crew(
    agents=[idea_generator_agent, writer_agent, evaluator_agent ],
    tasks=[generate_ideas, write_report, evaluate_report, rewrite_report],
    verbose=True,
    full_output=True
)

result = crew_startup.kickoff()
print("\n\n########################")
print("## Run Result:")
print("########################\n")
print(result)
