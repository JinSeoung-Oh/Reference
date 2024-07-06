## From https://generativeai.pub/llama-cpp-agent-007-87855676d7a6

from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

from rich.console import Console
console = Console(width=90)
import datetime
from time import sleep

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

#SetFilename
tstamp = datetime.datetime.now()
tstamp = str(tstamp).replace(' ','_')
tstamp = str(tstamp).replace(':','_')
logfile = f'{tstamp[:-7]}_log.txt'
sleep(2)
#Write in the history the first 2 sessions
writehistory(logfile,f'Created with 🌀 Qwen2-0.5b-instruct\n---\n\n\n')   

model = LlamaCppServerProvider("http://127.0.0.1:8080") 
#default port llamafile server is 8080
agent = LlamaCppAgent(
    model,
    system_prompt="",
)

product_description = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a product description writer",
    prompt="""Write a detailed product description for {product_name}, 
              including its features and benefits."""
)

product_usp = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a unique selling proposition (USP) creator",
    prompt="Create a compelling USP for {product_name} based on the 
            following product description:\n--\n{out_0}"
)

target_audience = AgentChainElement(
    output_identifier="out_2",
    system_prompt="You are a target audience identifier",
    prompt="Identify the target audience for {product_name} based 
            on the following product description and USP:\n--\n
            Product Description:\n{out_0}\nUSP:\n{out_1}"
)

marketing_channels = AgentChainElement(
    output_identifier="out_3",
    system_prompt="You are a marketing channel strategist",
    prompt="Suggest the most effective marketing channels to promote 
           {product_name} based on the following target audience:
           \n--\n{out_2}"
)

ad_copy = AgentChainElement(
    output_identifier="out_4",
    system_prompt="You are an advertising copywriter",
    prompt="Write engaging ad copy for {product_name} based on the 
           following product description, USP, and target audience:
           \n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\n
           Target Audience:\n{out_2}"
)

landing_page = AgentChainElement(
    output_identifier="out_5",
    system_prompt="You are a landing page designer",
    prompt="Create a high-converting landing page structure for {product_name} 
            based on the following product description, USP, target audience, 
            and ad copy:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\n
            Target Audience:\n{out_2}\nAd Copy:\n{out_4}"
)

email_campaign = AgentChainElement(
    output_identifier="out_6",
    system_prompt="You are an email marketing specialist",
    prompt="Develop an email campaign for {product_name} based on the 
            following product description, USP, target audience, and  
            landing page structure:\n--\nProduct Description:\n{out_0}\n
            USP:\n{out_1}\nTarget Audience:\n{out_2}\n
            Landing Page Structure:\n{out_5}"
)

social_media_posts = AgentChainElement(
    output_identifier="out_7",
    system_prompt="You are a social media content creator",
    prompt="Create a series of engaging social media posts for {product_name}  
            based on the following product description, USP, target audience, 
            and ad copy:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}
            \nTarget Audience:\n{out_2}\nAd Copy:\n{out_4}"
)

press_release = AgentChainElement(
    output_identifier="out_8",
    system_prompt="You are a press release writer",
    prompt="Write a compelling press release announcing the launch 
            of {product_name} based on the following product description, 
            USP, and target audience:\n--\nProduct Description:\n{out_0}
            \nUSP:\n{out_1}\nTarget Audience:\n{out_2}"
)

chain = [product_description, product_usp, target_audience, 
         marketing_channels, ad_copy, landing_page, email_campaign,
         social_media_posts, press_release, performance_metrics]

agent_chain = AgentChain(agent, chain)
# Ask for the input
productname = console.input(f'[bold green1]What do you want to launch> ') #ecxample "Smart Fitness Tracker"
start = datetime.datetime.now()
# RUn the Chain
with console.status("🌟 AI Assistant is working on it...",spinner='pong'):
    res = agent_chain.run_chain(additional_fields={"product_name": productname})
delta =  datetime.datetime.now() - start




