### From https://www.confident-ai.com/blog/llm-arena-as-a-judge-llm-evals-for-comparison-based-testing
### From https://www.marktechpost.com/2025/08/25/how-to-implement-the-llm-arena-as-a-judge-approach-to-evaluate-large-language-model-outputs/ 
### From https://github.com/Marktechpost/AI-Tutorial-Codes-Included/blob/main/LLM%20Evaluation/llm_arena_as_a_judge.py

"""
1. Problem Awareness
   Many users remain confused about the meaning and application of LLM evaluation metrics, which creates a barrier to entry when testing prompts
   and models.
   -a. Proposed Solution: LLM Arena-as-a-Judge — instead of complex metrics, use pairwise output comparison where the only decision is 
                          “which output is better?”

2. Overview of LLM Arena-as-a-Judge
   -a. Original Chatbot/LLM Arena: Built on Elo rating with human pairwise votes to rank models.
   -b. New Proposal: Replace the human judge with an LLM to automate and scale the process.
       -1. Users don’t select metrics.
       -2. They simply choose the better of two outputs — or let the LLM choose.
       -3. Can be directly applied to A/B testing of prompts, models, and app versions.
       -4. Implementable with DeepEval (open-source) in ~10 lines of code.

3. Limitations of the Original LLM Arena (Public Benchmark)
   -a. Strengths: Community-driven, useful for comparing many large models, leveraged in research and public leaderboards.
   -b. Limitations for internal workflows:
       -1. Cannot plug in your own app, prompt, or model.
       -2. Not suitable for iterative regression testing or CI/CD integration.
       -3. Not built for large-scale internal comparison of outputs.
       -4. → Motivation arises to replicate Arena-as-a-Judge for in-house evaluation.

4. Why Replicate/Adopt the Arena Approach?
   -a. Standard LLM-as-a-Judge (single-output scoring):
       -1. Requires designing multiple separate metrics (correctness, relevance, recall, etc.).
       -2. Evaluates outputs independently, making direct comparisons difficult.
   -b. Pairwise Comparison (Arena-as-a-Judge):
       -1. Decides relative superiority directly (“A vs B, which wins?”).
       -2. Simplifies and makes regression testing between prompt/model versions more intuitive.

5. Bias Mitigation Design
   -a. Blinding: Model names hidden during evaluation to avoid name bias.
   -b. Randomized positioning: Output order randomized to prevent left/right bias.
   -c. G-Eval Algorithm:
       -1. Two-step process: (1) generate evaluation steps from the criterion → (2) use those steps as instructions for the LLM judge.
       -2. Arena variant uses Chain-of-Thought (CoT) + form-filling approach specialized for winner selection.

6. Effectiveness, Limitations, and Usage Guidelines
   -a. Not a replacement. Regular LLM-as-a-Judge (single-output scoring):
       -1. Provides quantitative scores/graphs.
       -2. Suitable for online evaluation in production.
       -3. Allows fine-grained criteria and multi-turn evaluation.
   -b. Arena Approach:
       -1. Simple and intuitive setup, ideal for regression and A/B testing.
       -2. Less flexible for quantitative scoring or complex multi-turn scenarios.
   -c. Reliability:
       -1. Confident AI internal tests (250k+ cases) showed both standard G-Eval and Arena G-Eval align with human feedback at ~95%.

7. Selection Guide
   -a. Use Arena: For quick comparisons, deployment decisions, and regression testing.
   -b. Use Standard LLM-as-a-Judge: For quantitative metrics, fine-grained criteria, and production online monitoring.

8. Conclusion
   Arena G-Eval lowers the barrier of LLM evaluation by allowing judgments without metric design — simply choosing the “better output.”
   -a. Easily introduced via open-source DeepEval, applicable to any LLM.
   -b. Matches human agreement with ~95% reliability.
   -c. Practical for A/B testing and regression testing of prompts, models, and app versions.
   -d. Best strategy: use both approaches complementarily — Arena for fast comparisons, regular LLM-as-a-Judge for precise/production-level evaluations.

9. LLM-as-a-Judge vs. LLM Arena-as-a-Judge
   Category	| LLM-as-a-Judge (Conventional)	| LLM Arena-as-a-Judge (New Proposal)
   Evaluation Unit	| Single output (evaluated independently)	| Pairwise comparison (two outputs compared directly)
   Evaluation Metric	| Requires explicitly defined metrics (accuracy, relevance, coherence, etc.)	| No explicit metrics needed → only natural language criteria (e.g., “choose the friendlier response,” “pick the more accurate answer”)
   Judge	| LLM judge assigns a score (0–1)	| LLM judge selects the winner between two outputs
   Comparison Method	| Each output is scored independently → results compared afterward	| Outputs compared directly → immediate judgment
   Complexity |	Higher (requires metric design, management, and aggregation)	| Lower (simply choose a winner)
   Output Result	| Quantitative scores (e.g., 0.82 vs. 0.76)	| Qualitative results (e.g., “A wins” vs. “B wins” + reasoning)
   Use Cases	| Overall model performance evaluation, multi-turn dialogue analysis	| Prompt/app/model A/B testing, regression testing
   Advantages	| Provides quantitative metrics, Enables fine-grained, detailed analysis | Intuitive and fast setup, Supports automated A/B testing and large-scale comparisons, Matches human judgments with over 95% agreement
   Disadvantages	| Requires designing and managing multiple metrics, Direct A/B comparison is more complex	| Lacks quantitative scoring, Limited in multi-turn dialogue evaluation
"""

pip install deepeval google-genai openai

import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass('Enter OpenAI API Key: ')
os.environ['GOOGLE_API_KEY'] = getpass('Enter Google API Key: ')

from deepeval.test_case import ArenaTestCase, LLMTestCase, LLMTestCaseParams
from deepeval.metrics import ArenaGEval

context_email = """
Dear Support,
I ordered a wireless mouse last week, but I received a keyboard instead. 
Can you please resolve this as soon as possible?
Thank you,
John
"""

prompt = f"""
{context_email}
--------

Q: Write a response to the customer email above.
"""
==================================================================================
from openai import OpenAI
client = OpenAI()

def get_openai_response(prompt: str, model: str = "gpt-4.1") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

openAI_response = get_openai_response(prompt=prompt)
===================================================================================
from google import genai
client = genai.Client()

def get_gemini_response(prompt, model="gemini-2.5-pro"):
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text
geminiResponse = get_gemini_response(prompt=prompt)
====================================================================================
a_test_case = ArenaTestCase(
    contestants={
        "GPT-4": LLMTestCase(
            input="Write a response to the customer email above.",
            context=[context_email],
            actual_output=openAI_response,
        ),
        "Gemini": LLMTestCase(
            input="Write a response to the customer email above.",
            context=[context_email],
            actual_output=geminiResponse,
        ),
    },
)

metric = ArenaGEval(
    name="Support Email Quality",
    criteria=(
        "Select the response that best balances empathy, professionalism, and clarity. "
        "It should sound understanding, polite, and be succinct."
    ),
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model="gpt-5",  
    verbose_mode=True
)

metric.measure(a_test_case)
