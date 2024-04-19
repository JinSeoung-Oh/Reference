from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("env_variables.env")
client = OpenAI()

from promptrefiner import AbstractLLM, PromptTrackerClass, OpenaiCommunicator

class LlamaCPPModel(AbstractLLM):
    def __init__(self, base_url, api_key, temperature=0.1, max_tokens=200):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
    def predict(self, input_text, system_prompt):
        response = self.client.chat.completions.create(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},  # Update this as per your needs
            {"role": "user", "content": input_text}
        ],
        temperature=self.temperature,
        max_tokens=self.max_tokens,
    )
        llm_response = response.choices[0].message.content
        return llm_response


llamamodel = LlamaCPPModel(base_url="http://localhost:8000/v1", api_key="sk-xxx", temperature=0.1, max_tokens=400)
#########################################
input_evaluation_1 = """In an era where the digital expanse collided with the realm of human creativity, two figures stood at the forefront. Ada, with her prophetic vision, laid the groundwork for computational logic, while Banksy, shrouded in anonymity, painted the streets with social commentary. Amidst this, the debate over AI ethics, spearheaded by figures like Bostrom, questioned the trajectory of our technological companions. This period, marked by innovation and introspection, challenges us to ponder the relationship between creation and creator."""

input_evaluation_2 = """As the digital revolution accelerated, two contrasting visions emerged. Musk's endeavors to colonize Mars signified humanity's boundless ambition, while Hawking's warnings about AI highlighted the existential risks of unbridled technological advancement. Amid these towering ambitions and cautions, the resurgence of environmentalism, led by figures like Thunberg, reminded us of our responsibilities to Earth. This dichotomy between reaching for the stars and preserving our home planet defines the modern dilemma."""

output_evaluation_1 = """
["Ada Lovelace: Computational logic", "Banksy: Social commentary through art", "Nick Bostrom: AI ethics", "Digital era: Innovation and introspection", "AI ethics: The debate over the moral implications of artificial intelligence"]
"""

output_evaluation_2 = """
["Elon Musk: Colonization of Mars", "Stephen Hawking: Warnings about AI", "Greta Thunberg: Environmentalism", "Digital revolution: Technological advancement and existential risks", "Modern dilemma: Balancing ambition with environmental responsibility"]
"""

input_evaluations = [input_evaluation_1, input_evaluation_2]
output_evaluations = [output_evaluation_1, output_evaluation_2]
##########################################
init_sys_prompt = """You are an AI that receives an input text. Your task is to output a pythoning string where every strings is the name of a person with what they are associated with"""
promptTracker = PromptTrackerClass(init_system_prompt = init_sys_prompt)
promptTracker.add_evaluation_examples(input_evaluations, output_evaluations)

promptTracker.run_initial_prompt(llm_model=llamamodel)

print(promptTracker.llm_responses[0][0])
print(promptTracker.llm_responses[0][1])

"""
```python persons = {     "Musk, Elon": "space_exploration",     "Hawking, Stephen": "ai_ethics",
"Thunberg, Greta": "environmentalism" } print(f"Elon Musk is known for {persons['Musk, Elon']}")
print(f"Stephen Hawking is known for {persons['Hawking, Stephen']}") print(f"Greta Thunberg is known
for {persons['Thunberg, Greta']}") ``` This Python code creates a dictionary called `persons` where
each key-value pair represents a person and their associated field. The `print` statements then
display the person's name and their field of expertise.
"""

openai_communicator = OpenaiCommunicator(client=client, openai_model_code="gpt-4-0125-preview")
openai_communicator.refine_system_prompt(prompttracker=promptTracker, llm_model=llamamodel, number_of_iterations=3)

"""
1. We send the outcomes of previous rounds to the GPT-4 model. This includes the system prompts we’ve tested and the performance of our local LLM when applying these prompts to our evaluation texts.
2. Based on this information, GPT-4 proposes an improved system prompt.
3. We then apply this newly suggested prompt to our local LLM for another round of testing.
"""

print(promptTracker.llm_responses[1][1])
print(promptTracker.llm_system_prompts[1])

