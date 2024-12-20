### From https://python.langchain.com/v0.1/docs/modules/memory/agent_with_memory/

import os

from langchain.agents import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
os.environ["GOOGLE_CSE_ID"] = "GOOGLE_CSE_ID"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

prompt = hub.pull("hwchase17/react")
memory = ChatMessageHistory(session_id="test-session")

llm = OpenAI(temperature=0)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {"input": "How many people live in canada?"},
    config={"configurable": {"session_id": "<foo>"}},
)

