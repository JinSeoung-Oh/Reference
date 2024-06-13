""" Have to see this link for detail about LLM agents with langchain : https://levelup.gitconnected.com/implementation-of-llm-agents-should-you-opt-for-langchain-8e7fec937a58
    That link provide detail explain about each LLM agents functions
"""

import json
from openai import OpenAI, AzureOpenAI
import os

######## Begin of Helper Functions ########
class Search:
    def __init__(self):
        self.document = None

    def __call__(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        docstore = Wikipedia()
        document = docstore.search(term)
        self.document = document.page_content
        paragraphs = document.page_content.split("\n\n")
        return paragraphs[0]

def search(term):
        import wikipedia

        try:
            page_content = wikipedia.page(term).content
            result = page_content
        except wikipedia.PageError:
            result = f"Could not find [{term}]. Similar: {wikipedia.search(term)}"
        except wikipedia.DisambiguationError:
            result = f"Could not find [{term}]. Similar: {wikipedia.search(term)}"
        return result

class LookUp:
    def __init__(self):
        self.lookup_index = 0
        self.lookup_str = ""
        self.document = None

    def __call__(self, term: str, document: str) -> str:
        """Lookup a term in document (if saved). If found, return the lookup_index-th result."""
        paragraphs = document.page_content.split("\n\n")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

lookup = LookUp()

def finish_or_execute_tools(llm_message):
    if llm_message.function_call:
        
        # stringified arguments
        args= llm_message.function_call.arguments

        # parsed to a dictionary
        args = json.loads(args) # parse args

        # execute actions
        action = llm_message.function_call.name
        print(f'LM output: Call the {action} tool with the argument: \n {args}' )
        if action == 'Search':
            observation = search(**args)
        elif action == 'Lookup':
            observation = lookup(**args)
        else:
            observation = 'No such tool'
        print(f'{action} result: \n {observation}')
        return action, observation
    else:
        print('Sent to user:', llm_message.content)
        return None, llm_message.content
######## End of Helper Functions ########

######## Begin of Main Function ########
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],  
    api_version=os.environ['AZURE_OPENAI_API_VERSION'], 
)
openai_functions = [{
    'name': 'Search',
    'description': 'Search for a term in the docstore.',
    'parameters': {'properties': {
        'term': {'type': 'string'}},
        'required': ['term'],
        'type': 'object'}
    },
    {'name': 'Lookup',
     'description': 'Lookup a term in the docstore.',
     'parameters': {'properties': {
         'term': {'type': 'string'}},
         'required': ['term'],
         'type': 'object'}
}]

# user input
user_input = "3rd president in United State?" 
current_messages = [{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_input}]

# interactions between the user, GPT-4 and tools
loop = True
while loop:
    # GPT response
    result = client.chat.completions.create(
                model="gpt4",
                messages=current_messages,
                functions = openai_functions
            )
    llm_message = result.choices[0].message

    action_name, observation = finish_or_execute_tools(llm_message)

    # prepare new messages/prompt
    if action_name:
        # action: iteratively convert the object to dict
        action_dict = json.loads(json.dumps(llm_message, default=lambda o: o.__dict__))
        action_dict = {k: v for k, v in action_dict.items() if v is not None}
        # observation
        observation_dict = {'role': 'function', 'content': observation[:1000], 'name': 'Search'}
        # append to current messages
        current_messages.extend([action_dict, observation_dict])
    else:
        loop = False

print('Final result to user:', observation)
######## End of Main Function ########
