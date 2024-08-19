## From https://blog.stackademic.com/unleashing-the-power-of-collective-intelligence-the-mixture-of-agents-approach-in-large-language-330b4ae5eadf

"""
1. Introduction: The Challenge of Scaling LLMs
   Large language models (LLMs) like GPT-4 and LLaMA have transformed natural language processing, excelling in diverse tasks.
   However, their growing size and the associated costs make further scaling impractical. 
   The Mixture-of-Agents (MoA) methodology offers a solution by using multiple LLMs to leverage their collective strengths and improve performance. 
   Together.ai has implemented this approach in their Together MoA model, achieving a significant performance boost on benchmarks like AlpacaEval 2.0.

2. The Mixture-of-Agents Methodology
   The MoA approach involves a layered architecture where each layer contains several LLM agents. 
   These agents collaboratively process outputs from previous layers, refining responses iteratively until a final, robust output is produced. 
   This structure mimics the Mixture-of-Experts (MoE) approach, which involves multiple specialized expert networks working together. 
   The MoA method draws inspiration from MoE’s ability to harness diverse capabilities for complex problem-solving.

   - Key Mechanisms of MoA
     1) Collaborative Processing: LLMs can produce better outputs when referencing the outputs of other models, even if those auxiliary outputs are of lower quality.
                                  This phenomenon is central to the MoA strategy.
     2) Layered Structure: Each layer contains multiple agents that refine the input and pass it forward. The result is a comprehensive output after several iterations, 
                           combining the strengths of each model.

3. Benchmark Performance
   The MoA approach has been tested across multiple benchmarks:
   - AlpacaEval 2.0: Together MoA achieved a win rate of 65.1%, outperforming GPT-4 Omni (57.5%).
   - FLASK Evaluation: MoA models demonstrated superior robustness, correctness, efficiency, factuality, commonsense, insightfulness, and completeness, though they were noted to be slightly verbose.

4. Advantages of the MoA Approach
   1) Enhanced Performance
      By pooling the capabilities of multiple models, the MoA method generates higher-quality outputs.
   2) Cost-Effectiveness
      Using open-source models and optimizing layer and agent configurations allows for significant performance gains at a reduced cost.
   3) Scalability and Flexibility
      The MoA framework can be applied to a wide range of LLMs, making it adaptable to various architectures and tasks.

5. Together.ai’s Implementation of MoA
   Together.ai’s implementation employs a layered architecture where each layer consists of several LLM agents. 
   The setup balances quality and performance, with three layers refining responses through collaborative processing.

   - Proposers
     Six open-source models, including WizardLM-2–8x22b and Qwen1.5–110B-Chat, generate initial responses. 
     Each model contributes unique perspectives, enhancing the diversity of inputs.
   - Aggregators
     Qwen1.5–110B-Chat serves as the final aggregator, synthesizing inputs into a cohesive output. 
     The aggregator’s role is crucial for ensuring accuracy and insightfulness.

6. Variants of Together MoA
   - Together MoA
     The primary model using three layers and Qwen1.5–110B-Chat as the aggregator.
   - Together MoA-Lite
     A lightweight version with two layers and Qwen1.5–72B-Chat as the aggregator.
   - Together MoA w/ GPT-4o
     A variant that uses GPT-4o as the final aggregator, maintaining three layers.

7. Cost-Effectiveness of MoA
   The implementation is designed to balance performance and cost. The Pareto front in cost-performance analyses shows that MoA achieves an optimal tradeoff,
   making it highly efficient.
"""

import os
from together import Together
os.environ["TOGETHER_API_KEY"] = "your_api_key_here"

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
import asyncio
from together import AsyncTogether

async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

user_prompt = "What is Karma Yoga as per Bhagavad Gita, Vyadha Gita, Yoga Vasistham and Tripura Rahasya?"
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


async def run_llm(model):
    """Run a single LLM call with a reference model."""
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    print(f"Response from {model}: {response.choices[0].message.content}\n")
    return response.choices[0].message.content


async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggregator_system_prompt},
            {"role": "user", "content": ",".join(str(element) for element in results)},
        ],
        stream=True,
    )

    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


# asyncio.run(main())
await main()
