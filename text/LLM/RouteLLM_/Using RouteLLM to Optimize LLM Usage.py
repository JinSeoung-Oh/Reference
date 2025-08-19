### From https://www.marktechpost.com/2025/08/10/using-routellm-to-optimize-llm-usage/?amp

!pip install "routellm[serve,eval]"

import os
from getpass import getpass
os.environ['OPENAI_API_KEY'] = getpass('Enter OpenAI API Key: ')

----
!wget https://raw.githubusercontent.com/lm-sys/RouteLLM/main/config.example.yaml
----

import pandas as pd
from routellm.controller import Controller

client = Controller(
    routers=["mf"],  # Model Fusion router
    strong_model="gpt-5",       
    weak_model="o4-mini"     
)

----
!python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.1 --config config.example.yaml
----

threshold = 0.24034

prompts = [
    # Easy factual (likely weak model)
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the largest planet in our solar system?",
    
    # Medium reasoning (borderline cases)
    "If a train leaves at 3 PM and travels 60 km/h, how far will it travel by 6:30 PM?",
    "Explain why the sky appears blue during the day and red/orange during sunset.",
    
    # High complexity / creative (likely strong model)
    "Write a 6-line rap verse about climate change using internal rhyme.",
    "Summarize the differences between supervised, unsupervised, and reinforcement learning with examples.",
    
    # Code generation
    "Write a Python function to check if a given string is a palindrome, ignoring punctuation and spaces.",
    "Generate SQL to find the top 3 highest-paying customers from a 'sales' table."
]


win_rates = client.batch_calculate_win_rate(prompts=pd.Series(prompts), router="mf")

# Store results in DataFrame
_df = pd.DataFrame({
    "Prompt": prompts,
    "Win_Rate": win_rates
})

# Show full text without truncation
pd.set_option('display.max_colwidth', None)
