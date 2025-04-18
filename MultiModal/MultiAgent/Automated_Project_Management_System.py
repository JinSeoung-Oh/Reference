### From https://medium.com/the-ai-forum/building-an-automated-project-management-system-with-langgraph-supervisor-and-groq-e85f4c8ef41c

%pip install -qU langchain langchain-core langchain-groq langchain-community langchain-openai langgraph-supervisor langgraph

from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

from langchain_groq import ChatGroq
groq_model=ChatGroq(model="deepseek-r1-distill-llama-70b",temperature=0.6)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from IPython.display import Markdown,display

#1. Task planning Tool
#
#
def planningTool(project_type,project_requirements,team_mbers,objective):
  """Task Planner Function tool that helps to plan tasks"""
  prompt = """Carefully analyze the project requirements for the Project : {project_type} and break them down into individual tasks.
  Define each task's scope in detail,set achievable timelines and ensure that all the dependencies are accounted for:

  Project Objective: {objective}

  Project Requirement:
  {project_requirements}

  Team Members:
  {team_members}

  The final output should be a comprehensive list of taks with detailed scope,timelines,description,dependencies and deliverables.
  Your final OUTPUT must include a Gnatt Chart or similar timeline visualization specific to {project_type} project.
  """
  final_prompt = PromptTemplate(input_variables=["project_type","project_requirements","team_members","objective"],template=prompt)
  chain = final_prompt | groq_model | StrOutputParser()
  response = chain.invoke({"project_type":project_type,"project_requirements":project_requirements,"team_members":team_mbers,"objective":objective})
  return response

#2. Estimation Planning Tool
#
def estimationTool(project_type):
  """Resource and Time Estimation Function tool that helps to estimate time,resources and effort reuired to complete a project"""
  prompt = """Thoroughly evaluate each task  in the {project_type} to estimate time,resources and effort reuired to complete the project.
  Use hiostorical data,task,complexity and available resources to provide a realistic estimate.
  The Output should be a detailed estimation report outlining the time,resources and effort required for each task in the {project_type} project.
  Take into consideration the report from the Task Planner Function tool.The report MUST include a summary of any risks or uncertainties  with the estimations encountered during the estimation process.
  """
  final_prompt = PromptTemplate(input_variables=["project_type"],template=prompt)
  chain = final_prompt | groq_model | StrOutputParser()
  response = chain.invoke({"project_type":project_type})
  return response
#
# 3. Resource allocation Planning Tool
#
def resourceAllocationTool(project_type,industry):
  """Resource Allocation Function tool that helps to manage resources based on skills and availability"""
  prompt = """With a deep understanding of the team dynamics and resource management in {industry},you have a track record of
  ensuring that the right person is always assigned to the right task.Your startegic thinking ensures that the {project_type} project team
  is utilized to it's full potential without over burdening or over allocating any individuals.Optimize allocation task for the {project_type}
  by balancing team members,skills ,availability and current workload to maximize efficiency and productivity leading to project success.
  """
  final_prompt = PromptTemplate(input_variables=["project_type","industry"],template=prompt)
  chain = final_prompt | groq_model | StrOutputParser()
  response = chain.invoke({"project_type":project_type,"industry":industry})
  return response

task_planning_agent = create_react_agent(model=groq_model,
                                         tools=[planningTool],
                                         name="Task_Planner",
                                         prompt="""You are expert Task Planner""")
task_planning_agent

estimation_planning_agent = create_react_agent(model=groq_model,
                                         tools=[estimationTool],
                                         name="Task_estimation",
                                         prompt="""You are expert Task estimator proficient in estimating time,resouce and effort required to complete a project based on the Task Planner Function tool""")
estimation_planning_agent

resource_allocation_agent = create_react_agent(model=groq_model,
                                         tools=[resourceAllocationTool],
                                         name="Resource_allocation",
                                         prompt="""You are expert resource manager proficient in allocating tasks for Project by balancing team members,skills ,availability and current workload to maximize efficiency and productivity leading to project success.You need to take inputs from Planner Function tool and Resource and Time Estimation Function tool.""")

project_planning_workflow  = create_supervisor([task_planning_agent,estimation_planning_agent,resource_allocation_agent],
                                               model=groq_model,
                                               #output_mode="last_message",
                                               prompt=("You are a Veteran Project Manager managing Task Planning Estimation and allocation"
                                               "For Task Planning use task_planner_agent and produce detailed should be a comprehensive list of taks with detailed scope,timelines,description,dependencies and deliverables."
                                               "For Task estimation use estimation_planning_agent to produce estimation fror time,resources and effort required to complete the project"
                                               "For Resource allocation use resource_allocation_agent to perform task allocation ")
                                               )

app = project_planning_workflow.compile()

ask_planning_agent.invoke({"project_type":project_type,"project_requirements":project_requirements,"team_members":team_mbers,"objective":objective})
estimation_planning_agent.invoke({"project_type":project_type})
resource_allocation_agent.invoke({"project_type":project_type,"industry":industry})

prompt = f"""Based on the information provided below please preapare a detailed Project planning,estimation and allocation report

"industry":{industry},
"objective":{objective}
"project_type":{project_type},
"project_requirements":{project_requirements},
"team_members":{team_mbers}


The final output should have the following baserd on the outputs from the Task planner function tool,resourceestimator function tool and resource allocation function tool
1. The final output should be a comprehensive list of taks with detailed scope,timelines,description,dependencies and deliverables.
2. Your final OUTPUT SHOULD include a Gnatt Chart or similar timeline visualization specific to {project_type} project.
3. A detailed estimation report outlining the time,resources and effort required for each task in the {project_type} project.
4. The report MUST include a summary of any risks or uncertainties  with the estimations encountered during the estimation process based on the project requirements.
5. The report should also specify the task allocation details, as to how the team members are allocated to each task based on the:\n{team_mbers}
"""

result = app.invoke({"messages":"user","content":prompt})
display(Markdown(result['messages'][-1].content))
