# From https://gathnex.medium.com/connect-your-llm-to-the-internet-with-microsoft-autogen-3bc4c655e7c0

!pip install -q pyautogen~=0.1.0 docker openai 

import autogen
#Follow the same format for model and api arguments.
config_list = [
    {
        'model': 'gpt-3.5-turbo',
        'api_key': 'OpenAI API key'
    }
]
llm_config={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

user_proxy.initiate_chat(
    assistant,
    message="""
what is current time in Akureyri,Iceland ?
"""
)

