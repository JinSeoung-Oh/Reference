### https://medium.com/dataherald/high-accuracy-text-to-sql-with-langchain-840742133b83

! pip install --upgrade --quiet dataherald langchain langchain-core langchain-community langchain-openai

from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain_community.utilities.dataherald import DataheraldAPIWrapper
from langchain_community.tools.dataherald.tool import DataheraldTextToSQL
from langchain_core.prompts.prompt import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
import matplotlib.pyplot as plt
import numpy as np
import uuid
import ast
import json

# Setting the api keys in Google colab
openai_api_key = userdata.get('OPENAI_API_KEY')
dataherald_api_key = userdata.get('DATAHERALD_API_KEY')

api_wrapper = DataheraldAPIWrapper(dataherald_api_key= dataherald_api_key,db_connection_id="661fd247412d933d48439ebc")

def clean_string(s):
    # Remove all spaces and newlines, then convert to lowercase
    return s.replace(" ", "").replace("\n", "").lower()

# the text to sql engine
text_to_sql_tool = DataheraldTextToSQL(api_wrapper=api_wrapper)

# execute sql query tool
def execute_sql_query(sql_query: str) -> str:
  generated_queries = api_wrapper.dataherald_client.sql_generations.list(page=0,
                               page_size=20,
                               order='created_at',
                               ascend=False)
  query_id = ""
  for query in generated_queries:
    if clean_string(query.sql) == clean_string(sql_query):
      query_id = query.id
      break
  if not query_id:
    raise Exception("Query has not found")
  query_results = api_wrapper.dataherald_client.sql_generations.execute(id=query_id)
  return str(query_results)

execute_query_tool = StructuredTool.from_function(
    func=execute_sql_query,
    name="Execute Query",
    description="Usefull for executing a SQL query over the database, input is a sql query generated by the text-to-sql tool",
)

# A function to plot a list
def plot_and_save_array(dict_of_values):
  dict_of_values = json.loads(dict_of_values.strip().replace("'", '"'))
  items = list(dict_of_values.keys())
  values = list(dict_of_values.values())
  plt.figure(figsize=(5, 3))
  plt.plot(items, values, marker='o')
  plt.title("Array Plot")
  plt.xlabel("Items")
  plt.ylabel("Values")
  plt.xticks(rotation=45)
  plt.grid(True)
  identifier = uuid.uuid4().hex[:6] + ".png"
  plt.savefig(identifier)
  plt.show()
  return "success"

plotting_tool = StructuredTool.from_function(
    func=plot_and_save_array,
    name="Plotting Results",
    description="A tool which receives a valid json object with keys being the x axes values and for each key we should have a single value",
)

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with question and analysis on the database values.

Assistant plan:
1) Use the text-to-sql tool to generated a SQL query for the user question
2) Execute the generated SQL query over the database using the Execute Query tool
3) Use Plotting Results tool to plot the results that are returned by the Execute Query tool

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo-preview", temperature=0)
tools = [text_to_sql_tool, execute_query_tool, plotting_tool]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {"input": "What was the rent price in LA for the summer and fall 2022 for each month? Plot the results"}
)







