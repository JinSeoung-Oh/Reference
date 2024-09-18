### https://medium.com/@datadrifters/worlds-top-llm-is-now-open-source-reflection-llama-3-1-70b-beats-gpt-4o-and-claude-3-5-sonnet-d74b913ce8ba

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator, TextGeneration

REASONING_PROMPT = (
    "You are an AI assistant specialized in logical thinking and problem-solving. Your"
    " purpose is to help users work through complex ideas, analyze situations, and draw"
    " conclusions based on given information. Approach each query with structured thinking,"
    " break down problems into manageable parts, and guide users through the reasoning"
    " process step-by-step."
)

REFLECTION_SYSTEM_PROMPT = """
You're an AI assistant that responds the user with maximum accuracy. To do so, your first will think what the user is asking for, thinking step by step. During this thinking phase, you will have reflections that will help you clarifying ambiguities. In each reflection you will list the possibilities and finally choose one. Between reflections, you can think again. At the end of the thinking, you must draw a conclusion. You only need to generate the minimum text that will help you generating a better output, don't be verbose while thinking. Finally, you will generate an output based on the previous thinking.
This is the output format you have to follow:
```
<thinking>
Here you will think about what the user asked for.
<reflection>
This is a reflection.
</reflection>
<reflection>
This is another reflection.
</reflection>
</thinking>
<output>
Here you will include the output
</output>
```
""".lstrip()

with Pipeline(name="reflection-tuning") as pipeline:
    generate_instructions = MagpieGenerator(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            magpie_pre_query_template="llama3",
            generation_kwargs={"temperature": 0.8, "max_new_tokens": 2048},
        ),
        system_prompt=REASONING_PROMPT,
        batch_size=5,
        num_rows=5,
        only_instruction=True,
    )

    response = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={"temperature": 0.8, "max_new_tokens": 2048},
        ),
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        input_batch_size=5,
    )

    generate_instructions >> response

if __name__ == "__main__":
    distiset = pipeline.run()
    distiset.push_to_hub(
        "gabrielmbmb/distilabel-reflection-tuning", include_script=True
    )
