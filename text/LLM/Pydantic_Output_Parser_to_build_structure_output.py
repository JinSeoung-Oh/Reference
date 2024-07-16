## From https://generativeai.pub/transform-unstructured-llm-output-into-structured-data-with-output-parsing-pydantic-models-a-2f699b718a6b

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

class SuggestedChart(BaseModel):
        main_chart_type: str = Field(description="The suggested type of main category of chart to use")
        sub_chart_type: str = Field(description="The sub chart to use within the main category")
        column1: str = Field(description="The first column that will be plotted in the chart")
        column2: str = Field(description="The second column that will be plotted in the chart")
        column3: str = Field(description="The third column that will be plotted in the chart")
        aggregation: str = Field(description="The data aggregation used within the plot")
        chart_title: str = Field(description="Main title of the chart")
        description: str = Field(description="Description of what the chart shows")

class Charts(BaseModel):
        chart_1: SuggestedChart = Field(description="The first chart suggested")
        chart_2: SuggestedChart = Field(description="The second chart suggested")
        chart_3: SuggestedChart = Field(description="The third chart suggested")

parser = PydanticOutputParser(pydantic_object=Charts)

parser.get_format_instructions()

prompt_template = """
Assistants main role is to analyse a schema of a dataset
and suggest to the human types of data charts to use
to visualise the data.

Here are your instructions

INSTRUCTIONS
------------

1. Analyse the data types within the schema
2. Suggest 3 DIFFERENT main chart types
3. Suggest 1 sub chart types from each main category
4. Suggest the columns to be plotted in the suggested sub chart
5. Suggest the aggregations needed for each sub chart. For example; sum, mean, etc.
6. Create a short description of what the chart shows

The chart options are structured as 

CHART STRUCTURE 

------------

main_chart_type;
  sub_chart_type1
  sub_chart_type2

You have the following charts to choose from

    scatter:
      basic
    pie:
      basic
      rose
      ring
    bar:
      stacked
      basic
      multi
    line:
      multi
      basic
      stacked
    area:
      multi
      basic
      stacked

{format_output_instructions} 

New Input: {input}  

"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input"]
    )

new_input_prompt = """

 Create the chart information for a dataset with schema and data types of

  SCHEMA
  ------

  {schema}

  DATA TYPES
  -------

  {data_types}

  Ensure to follow the given instructions step by step.

""".format(schema=metadata['Schema'], data_types=metadata["Data Types"])

gpt4o = ChatOpenAI(temperature=0.0, model="gpt-4o", openai_api_key="") #INSERT OPENAI API KEY HERE
chain = prompt | gpt4o | parser
chain_out = chain.invoke({"input": new_input_prompt, 
                         "format_output_instructions": parser.get_format_instructions()})

## Retrun 
"""
Charts(
chart_1=SuggestedChart(main_chart_type='bar', sub_chart_type='stacked', column1='province', column2='price', column3='variety', aggregation='average', chart_title='Average Price of Wine by Province and Variety', description='This chart shows the average price of wine categorized by province and variety. It helps to identify which provinces and varieties have higher or lower average prices.'), 
chart_2=SuggestedChart(main_chart_type='scatter', sub_chart_type='basic', column1='points', column2='price', column3='', aggregation='none', chart_title='Scatter Plot of Wine Points vs Price', description='This scatter plot visualizes the relationship between wine points and price. It helps to identify any correlation between the quality (points) and the cost (price) of the wines.'), 
chart_3=SuggestedChart(main_chart_type='pie', sub_chart_type='rose', column1='variety', column2='price', column3='', aggregation='sum', chart_title='Total Price Distribution by Wine Variety', description='This rose pie chart shows the total price distribution of wines by variety. It helps to understand which wine varieties contribute the most to the total price.')
)
"""
chain_out.chart_1
