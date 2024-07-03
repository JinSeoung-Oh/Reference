## From https://ai.gopubby.com/how-to-build-a-react-ai-agent-with-claude-3-5-and-python-95423f798640

prompt = """
You run in a loop of Thought, Action, Observation, Answer.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.
Answer will be the result of analysing the Observation

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

wikipedia:
e.g. wikipedia: Django
Returns a summary from searching Wikipedia

Always look things up on Wikipedia if you have the opportunity to do so.

Example session:

Question: What is the capital of France?

Thought: I should look up France on Wikipedia

Action: wikipedia: France

You should then call the appropriate action and determine the answer from 
the result

You then output:

Answer: The capital of France is Paris
"""

import anthropic
import re
import httpx

client = anthropic.Anthropic(api_key="YOUR API KEY HERE")

class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return message

    def execute(self):
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system = self.system,
            messages = self.messages
        )
        return message.content

action_re = re.compile('^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    i = 0
    bot = ChatBot(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return bot.messages

def wikipedia(q):
    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]

def calculate(what):
    return eval(what)

known_actions = {
    "wikipedia": wikipedia,
    "calculate": calculate
}

def get_last_message():
    for m in bot.messages[-1]['content'][0].text.split('\n'):
        print(m)


## Example
query("What is 20 * 15")
get_last_message()

query("What languages are spoken in Spain that are also spoken in France")
get_last_message()

