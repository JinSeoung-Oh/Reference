## From https://generativeai.pub/ai-agents-frameworks-bts-51965779f599

import ollama
import subprocess
subprocess.Popen(["ollama", "serve"])  # skip if Ollama is already running locally

tools = """
        (1) search_wikipeida: [keyword] - this tool searches wikipedia for the keyword and provides a summary
        (2) calculate: [expression]- this tool solves a simple BODMAS equation
"""

react_prompt = f"""
You will go through a loop of Thought, Action, and Observation to answer questions. 
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.

Your available actions are: 
{tools}

Continue this loop until you can conclude with a clear answer. 
Do note that you do not have any knowledge from wikipedia, and always prefer to use the search_tool incase you are unware of the keywords. 

You can only give (Thought and Action) or (Observation) or (Thought and Answer)

Some types of thoughts :
1. Thoughts that break the question down into components...
2. Thoughts that gauge the quality of observations...
3. Thoughts that critically questions and provide clarity...
4. Thoughts that carefully assess the keyword being passed to search and the ability of the keyword to be found in wikipedia... 

Examples :
{react_examples}

Do not limit to only two thoughts, you are free to have mutliple iteration of thoughts. The examples above are limited, and only show some of the approaches.
"""

class ReActAgent:
    
    def __init__(self, model=model, system_message=""):
        self.model = model
        self.memory = [{"role": "system", "content": system_message}]
    
    def add_memory(self, role, content):
        self.memory.append({"role": role, "content": content})
    
    def query_model(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        response = response['message']['content']
        return response

def get_wikipedia_html(keyword):
    # Step 1: Search for the best matching title using the Wikipedia API's opensearch endpoint
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "opensearch",
        "format": "json",
        "search": keyword,
        "limit": 1  # Get the best match only
    }

    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()
    
    # Check if there's a result
    if len(search_data[1]) == 0:
        return "No match found on Wikipedia for this keyword."

    # Step 2: Get the title of the best match and load the page summary
    best_match_title = search_data[1][0]
    
    print(best_match_title)

    # Step 3: Fetch the page content (summary) for the best match title
    page_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + best_match_title
    page_response = requests.get(page_url)
    page_data = page_response.json()

    # Check if the page has a summary
    if 'extract' in page_data:
        return page_data['extract']
    else:
        return "Summary not available for this topic."

def calculator(equation):
    try:
        return f"Result for calculation: {eval(equation)}'"
    except Exception as e:
        return f"Calculation error: {e}"

class ReActAgent:
    
    def __init__(self, model=model, system_message=""):
        self.model = model
        self.memory = [{"role": "system", "content": system_message}]
    
    def add_memory(self, role, content):
        self.memory.append({"role": role, "content": content})
    
    def query_model(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        response = response['message']['content']
        return response 
    
    def react_cycle(self, query, debug_flag=True):
        """Run the ReAct loop until an answer is found."""
        self.add_memory("user", query)
        response = ""
        i = 1 
        while "Answer:" not in response:
            response = self.query_model(self.memory)
            self.add_memory("assistant", response)
            
            # Parse Thought and Action in a single response
            thought, action_type, action_input = self.parse_thought_action(response)
            if debug_flag : print(f"\n\tNB===>thought:{thought}, action_type:{action_type}, action_input:{action_input}")
            if thought:
                if debug_flag : print("thinking.....")
                print(f"Thought {i}:", thought)
            if action_type:
                if debug_flag : print("action.....")
                print(f"Action {i}:",action_type,": ",action_input)
                observation = self.execute_action(action_type, action_input)
                print(f"Observation {i}:",observation[:300]) #only printing 300 characters
                self.add_memory("assistant", f"Observation: {observation}")
            if "Answer:" in response:
                answer = self.parse_answer(response)
                print(answer.group(0))
                return answer.group(0)
            i = i+1
            
    def parse_answer(self, response):
        """Parse the Answer from the response."""
        answer_match = re.search(r"Answer: (.*)", response)
        return answer_match
    
    def parse_thought_action(self, response):
        """Parse the Thought and Action parts from the response."""
        thought_match = re.search(r"Thought: (.*?)\n", response)
        action_match = re.search(r"Action: (\w+): (.*)", response)
        thought = thought_match.group(1) if thought_match else None
        action_type = action_match.group(1) if action_match else None
        action_input = action_match.group(2) if action_match else None
        return thought, action_type, action_input

    def execute_action(self, action_type, action_input):
        if action_type == "search_wikipedia":
            search_output = get_wikipedia_html(action_input)
            return f"Result for search on '{action_input}':\n'{search_output}'"
        elif action_type == "calculate":
            return calculator(equation=action_input)
        else:
            return "Unvailable Action Selected"
