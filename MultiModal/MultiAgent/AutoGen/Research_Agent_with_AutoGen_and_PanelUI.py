### From https://generativeai.pub/creating-a-research-agent-with-autogen-and-panel-ui-7386ebf76fd9

conda create -n research_agent_env python=3.9 -y
conda activate research_agent_env

python -m venv research_agent_env
source research_agent_env/bin/activate  
# On Windows: research_agent_env\Scripts\activate

pip install autogen panel 

import autogen
import panel as pn

# Configuration for the LLM model
model_configurations = [
    {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        'api_key': 'ollama',
    },
]

llm_settings = {"config_list": model_configurations, "temperature": 0, "seed": 53}

# Define the UserProxyAgent (Admin)
admin_agent = autogen.UserProxyAgent(
    name="Admin",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("exit"),
    system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin. 
    Only say APPROVED in most cases, and say EXIT when nothing to be done further. Do not say others.""",
    code_execution_config=False,
    default_auto_reply="Approved",
    human_input_mode="NEVER",
    llm_config=llm_settings,
)

# Define the AssistantAgent (Engineer)
engineer_agent = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_settings,
    system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
    Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
    ''',
)

# Define the AssistantAgent (Scientist)
scientist_agent = autogen.AssistantAgent(
    name="Scientist",
    llm_config=llm_settings,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code."""
)

# Define the AssistantAgent (Planner)
planner_agent = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
    The plan may involve an engineer who can write code and a scientist who doesn't write code.
    Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
    ''',
    llm_config=llm_settings,
)

# Define the UserProxyAgent (Executor)
executor_agent = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
)

# Define the AssistantAgent (Critic)
critic_agent = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=llm_settings,
)

# Create a GroupChat with all agents
group_chat = autogen.GroupChat(agents=[admin_agent, engineer_agent, scientist_agent, planner_agent, executor_agent, critic_agent], messages=[], max_round=50)
chat_manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_settings)

def print_messages(recipient, messages, sender, config):
    """
    Prints and sends the latest message from the sender to the recipient using the chat interface.
    Args:
        recipient (object): The recipient object containing recipient details.
        messages (list): A list of message dictionaries, where each dictionary contains message details.
        sender (object): The sender object containing sender details.
        config (dict): Configuration dictionary for additional settings.
    Returns:
        tuple: A tuple containing a boolean and None. The boolean is always False to ensure the agent communication flow continues.
    Notes:
        - The function prints the details of the latest message.
        - If the latest message contains the key 'name', it sends the message using the name and avatar from the message.
        - If the 'name' key is missing, it sends the message using a default user 'SecretGuy' and a ninja avatar.
    """
    print(f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}")
    
    if all(key in messages[-1] for key in ['name']):
        chat_interface.send(messages[-1]['content'], user=messages[-1]['name'], avatar=agent_avatars[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(messages[-1]['content'], user='SecretGuy', avatar='🥷', respond=False)

    return False, None  # required to ensure the agent communication flow continues

# Register the print_messages function as a reply handler for each agent
admin_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)

engineer_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 
scientist_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 
planner_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)

executor_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 
critic_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 

# Initialize Panel extension with material design
pn.extension(design="material")

def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # Initiate chat with the admin_agent
    admin_agent.initiate_chat(chat_manager, message=contents)

# Create a chat interface and send an initial message
chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("Send a message!", user="System", respond=False)
chat_interface.servable()



