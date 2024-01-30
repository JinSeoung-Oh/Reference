from langchain.llms import Ollama
llm_ollama = Ollama(model="YOUR_MODEL_NAME")

Insure_agent = Agent(
  role='Insure_agent',
  goal="""responsible for listing the travel plan from advisor and giving the short 
    insurance items based on the travel plan""",
  backstory="""You are an Insure agent who gives 
    the short insurance items based on the travel plan. 
    Don't ask questions. Make your response short.""",
  verbose=True,
  allow_delegation=False,
  llm=llm_ollama,
