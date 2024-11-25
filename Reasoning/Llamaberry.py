### From https://medium.com/@odhitom09/llamaberry-unlocking-advanced-chain-of-thought-in-ai-reasoning-85ac71f0e839

## Step 1: Setting the Stage
initial_system_prompt = """You are an AI assistant capable of detailed, step-by-step thinking. When presented with a question or problem, break down your thought process into clear, logical steps. For each step, explain your reasoning. Conclude with a final answer. Use the following markdown structure:

## Reasoning
1. [First step]
   **Explanation:** [Detailed explanation of this step]
2. [Second step]
   **Explanation:** [Detailed explanation of this step]
...

## Answer
[Final answer]

Be comprehensive and show your reasoning clearly."""

-----------------------------------------------------------------------------------------------------------------
## Step 2: The Thinking Process
async def generate_turn(query: str, previous_turns: list = None) -> str:
    is_first_turn = previous_turns is None or len(previous_turns) == 0
    if is_first_turn:
        messages = [{
            "role": "system",
            "content": initial_system_prompt
        }, {
            "role": "user",
            "content": query
        }]
    else:
        previous_content = "\n\n".join(previous_turns)
        messages = [{
            "role": "system",
            "content": followup_system_prompt
        }, {
            "role":
            "user",
            "content":
            f"Original Query: {query}\n\nPrevious Turns:\n{previous_content}\n\nProvide the next turn of reasoning."
        }]

    return await call_llm(messages)
-----------------------------------------------------------------------------------------------------------------
## Step 3: Putting It All Together
async def synthesize_turns(query: str, turns: list) -> str:
    turns_text = "\n\n".join(
        [f"Turn {i+1}:\n{turn}" for i, turn in enumerate(turns)])
    messages = [{
        "role": "system",
        "content": synthesis_prompt
    }, {
        "role":
        "user",
        "content":
        f"Original Query: {query}\n\nTurns of Reasoning:\n{turns_text}"
    }]
    return await call_llm(messages)

