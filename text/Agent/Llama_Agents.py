### From https://levelup.gitconnected.com/exploring-llama-agents-strengths-weaknesses-and-code-walkthrough-2a4a96602034

!pip install llama-agents llama-index-agent-openai

from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    PipelineOrchestrator,
    ServiceComponent,
    LocalLauncher,
    AgentOrchestrator,
    HumanService,
)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

def get_a_topic() -> str:
    """Returns a topic name."""
    return "The topic is: gpt-5."

def get_a_view() -> str:
    """Returns the view."""
    return "negative"

message_queue = SimpleMessageQueue()
tool = FunctionTool.from_defaults(fn=get_a_topic)

worker1 = FunctionCallingAgentWorker.from_tools([tool], llm=OpenAI(model="gpt-4o"))
agent1 = worker1.as_agent()

agent1_server = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for getting the topic.",
    service_name="topic_agent",
)

tool2 = FunctionTool.from_defaults(fn=get_a_view)

agent2 = OpenAIAgent.from_tools(
    [tool2],
    system_prompt="Get a view of positive or negative from perform the task tool.",
    llm=OpenAI(model="gpt-4o"),
)

agent2_server = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for getting view of positive or negative.",
    service_name="view_agent",
)

############ Set up the orchestrator ############
agent1_component = ServiceComponent.from_service_definition(agent1_server)
agent2_component = ServiceComponent.from_service_definition(agent2_server)

pipeline = QueryPipeline(chain=[agent1_component, agent2_component])

pipeline_orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(message_queue, pipeline_orchestrator)

launcher = LocalLauncher([agent1_server, agent2_server], control_plane, message_queue)
result = launcher.launch_single("What is your view of the topic?")

print(f"Result: {result}")

############ Agents with Auto Orchestration ############
orchestrator =AgentOrchestrator(llm=OpenAI(mode="gpt-4o"))
control_plane = ControlPlaneServer(message_queue, orchestrator)
result = launcher.launch_single("""Please answer: What is your topic? 
                    And what is your view of it?""")
orchestrator =AgentOrchestrator(llm=OpenAI())

############ Agents with Human Interaction ############
from llama_agents import HumanService

human_service = HumanService(
    service_name="Human_Service", description="Answer question about the topic.", 
      message_queue=message_queue,
)

agent_human_component = ServiceComponent.from_service_definition(human_service)

pipeline = QueryPipeline(chain=[agent_human_component, agent2_component, ])

pipeline_orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(message_queue, pipeline_orchestrator)

launcher = LocalLauncher([human_service, agent2_server], control_plane, message_queue)
result = launcher.launch_single("What is the topic and what is the view of this topic?")


import nest_asyncio

nest_asyncio.apply()








