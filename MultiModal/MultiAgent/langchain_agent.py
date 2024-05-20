## From https://medium.com/@guimaraesabri/building-an-ai-agent-with-langchain-lcel-and-langchain-hub-by-langsmith-7956eba24911

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

model = ChatOpenAI(api_key="...")

prompt_template = PromptTemplate.from_template(
    """"
    You are an expert in {theme}. 
    Your mission is to provide in-depth content on the topic, 
    answer questions, and act as an assistant. 
    Use emojis to make your answers more engaging and friendly. 
    Always strive to be approachable and helpful, offering the 
    most accurate and useful information possible to users.
    """
)
data_engineering_prompt_template = prompt_template.format(theme="Data Engineering")
mlops_prompt_template = prompt_template.format(theme="MLOps")
ml_engineering_prompt_template = prompt_template.format(theme="Machine Learning Engineering")
data_science_prompt_template = prompt_template.format(theme="Data Science")

@tool
def chain_data_engineering(input_text) -> str:
    """
    Designed as an expert in data engineering, 
    this tool offers guidance on efficient data management. 
    It answers questions and assists in developing optimized data pipelines, 
    facilitating effective data collection, integration, and processing
    """
    prompt = ChatPromptTemplate.from_messages([
            ("system", data_engineering_prompt_template),
            ("user", "{input}")
        ])
    chain = prompt | model 
    return chain.invoke({"input": input_text})
    
    ...
    @tool
def chain_ml_engineering(input_text) -> str:
    """
    This tool specializes in machine learning engineering, 
    supporting you from model design to evaluation. 
    It answers questions about algorithm selection, hyperparameter tuning, 
    and more, ensuring the development of efficient and effective ML solutions.
    """
    prompt = ChatPromptTemplate.from_messages([
            ("system", ml_engineering_prompt_template),
            ("user", "{input}")
        ])
    chain = prompt | model 
    return chain.invoke({"input": input_text})
    ...

prompt = hub.pull("guimaraesabri/datacrafter")
tools = [chain_data_engineering, chain_ml_engineering, chain_mlops, chain_data_science]

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

### with FastAPI
app = FastAPI()

class Input(BaseModel):
    input_text: str
class Output(BaseModel):
    model_response: str

@app.post("/agent/invoke", response_model=Output)
def invoke_agent(input_data: Input):
    try:
        response = agent_executor.invoke({"input": input_data.input_text})
        return Output(model_response=response['output'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



