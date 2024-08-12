## From https://blog.stackademic.com/storm-stanfords-revolutionary-research-tool-harnessing-the-power-of-agents-and-agentic-workflows-a2fa0e1a7fe3
## https://github.com/stanford-oval/storm/tree/NAACL-2024-code-backup?source=post_page-----a2fa0e1a7fe3--------------------------------

"""
1. Overview of STORM
   STORM (Synthesis of Topic Outlines through Retrieval and Multiperspective question asking) is a pioneering AI tool developed by Stanford University,
   designed to revolutionize research and content creation by producing comprehensive, Wikipedia-style articles. 
   It emphasizes transparency by referencing the sources of its information, adding credibility and reliability.

2. Key Features of STORM
   -1. Comprehensive Content Creation
       STORM generates detailed and well-structured articles across various topics.
   -2. Local Runtime Capability
       Users can run STORM on their local machines, ensuring privacy and control over the research process.
   -3. Source Referencing
       Each piece of information is linked back to its original source, facilitating easy fact-checking and exploration.
   -4. Multi-Agent Research
       Utilizes a team of AI agents to conduct thorough research.
   -5. Open-Source Availability
       Being open-source, STORM is accessible globally, encouraging collaboration and continuous improvement.
   -6. Top-Down Writing Approach
       STORM uses a top-down strategy, focusing on creating an outline before filling in content, which helps convey information effectively.
   -7. Diverse Perspective Discovery
       Incorporates various perspectives for a more comprehensive understanding of topics.
   -8. Multi-Perspective Question Asking
       Simulates conversations with diverse viewpoints to explore topics deeply.

3. Understanding Agentic Systems
   Agentic systems, which are foundational to STORM, are AI frameworks that mimic human-like autonomy and intelligence. 
   They are capable of perceiving their environment, making decisions, taking actions, and learning over time. 
   These systems are characterized by:
   
   -1. Perception: Gathering and processing information from diverse sources.
   -2. Decision-Making: Determining the best course of action based on gathered data.
   -3. Action-Taking: Executing tasks or producing outputs.
   -4. Learning and Adapting: Improving performance over time.

4. Agentic Workflows involve a series of steps that these systems follow to complete tasks, including information retrieval,
   multi-perspective question asking, and content synthesis. STORM employs AI agents to execute these workflows, including:

   -1. Research Agents: Gather information from various sources.
   -2. Question-Asking Agents: Generate questions from different perspectives.
   -3. Expert Agents: Provide answers to these questions.
   -4. Synthesis Agents: Compile and organize information into a coherent article structure.

5. How STORM Works
   The process can be broken down into three main steps:
   -1. Retrieval: STORM deploys AI agents to search for relevant information across the internet.
   -2. Multi-Perspective Question Asking: Simulates interactions where different perspectives generate insightful questions, enhancing topic exploration.
   -3. Synthesis: Organizes and structures gathered information into a coherent article, ensuring logical flow and accessibility.

   For example, when tasked with creating content on Karma Yoga, STORM generated a comprehensive wiki covering aspects like spiritual texts,
   historical context, practices, and more.

6. Benefits of STORM’s Approach
   -1. Well-Structured Content: By using outlines before writing, STORM ensures articles are logically organized.
   -2. Comprehensive Coverage: Multi-perspective questions lead to thorough exploration of topics.
   -3. Diverse Viewpoints: Incorporates varied perspectives for balanced content.
   -4. Time-Saving: Automates pre-writing stages, reducing time and effort.
   -5. Consistency: Ensures consistent quality across topics.
"""

!pip install knowledge-storm

### With OpenAI
import os
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM

os.environ["OPENAI_API_KEY"]="your_api_key_here"
os.environ["YDC_API_KEY"]="your_YDC_api_key_here"
lm_configs = STORMWikiLMConfigs()
openai_kwargs = {
 'api_key': os.getenv("OPENAI_API_KEY"),
 'temperature': 1.0,
 'top_p': 0.9,
}

gpt_35 = OpenAIModel(model='gpt-4o-mini', max_tokens=500, **openai_kwargs)
gpt_4 = OpenAIModel(model='gpt-4o-mini', max_tokens=3000, **openai_kwargs)
lm_configs.set_conv_simulator_lm(gpt_35)
lm_configs.set_question_asker_lm(gpt_35)
lm_configs.set_outline_gen_lm(gpt_4)
lm_configs.set_article_gen_lm(gpt_4)
lm_configs.set_article_polish_lm(gpt_4)

engine_args = STORMWikiRunnerArguments(output_dir='output')
rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=engine_args.search_top_k)
runner = STORMWikiRunner(engine_args, lm_configs, rm)

topic = input('Topic: ')
runner.run(
 topic=topic,
 do_research=True,
 do_generate_outline=True,
 do_generate_article=True,
 do_polish_article=True,
)
runner.post_run()
runner.summary()

### With Claude AI
!pip install anthropic
!pip install knowledge-storm

import os
from argparse import ArgumentParser
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import ClaudeModel
from knowledge_storm.rm import YouRM, BingSearch
from knowledge_storm.utils import load_api_key

os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_key_here"
os.environ["YDC_API_KEY"] = "your_YDC_api_key_here"

lm_configs = STORMWikiLMConfigs()
claude_kwargs = {
 'api_key': os.getenv("ANTHROPIC_API_KEY"),
 'temperature': 1.0,
 'top_p': 0.9
}
conv_simulator_lm = ClaudeModel(model='claude-3-haiku-20240307', max_tokens=500, **claude_kwargs)
question_asker_lm = ClaudeModel(model='claude-3-sonnet-20240229', max_tokens=500, **claude_kwargs)
outline_gen_lm = ClaudeModel(model='claude-3-opus-20240229', max_tokens=400, **claude_kwargs)
article_gen_lm = ClaudeModel(model='claude-3–5-sonnet-20240620', max_tokens=700, **claude_kwargs)
article_polish_lm = ClaudeModel(model='claude-3–5-sonnet-20240620', max_tokens=4000, **claude_kwargs)
lm_configs.set_conv_simulator_lm(conv_simulator_lm)
lm_configs.set_question_asker_lm(question_asker_lm)
lm_configs.set_outline_gen_lm(outline_gen_lm)
lm_configs.set_article_gen_lm(article_gen_lm)
lm_configs.set_article_polish_lm(article_polish_lm)

engine_args = STORMWikiRunnerArguments(
 output_dir=args.output_dir,
 max_conv_turn=args.max_conv_turn,
 max_perspective=args.max_perspective,
 search_top_k=args.search_top_k,
 max_thread_num=args.max_thread_num,
)

if args.retriever == 'bing':
 rm = BingSearch(bing_search_api=os.getenv('BING_SEARCH_API_KEY'), k=engine_args.search_top_k)
elif args.retriever == 'you':
 rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=engine_args.search_top_k)

runner = STORMWikiRunner(engine_args, lm_configs, rm)
topic = input('Topic: ')
runner.run(
 topic=topic,
 do_research=args.do_research,
 do_generate_outline=args.do_generate_outline,
 do_generate_article=args.do_generate_article,
 do_polish_article=args.do_polish_article,
)
runner.post_run()
runner.summary()
