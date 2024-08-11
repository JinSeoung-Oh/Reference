## From https://towardsdatascience.com/serve-multiple-lora-adapters-with-vllm-5323b0425b82

!pip install vllm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download
from openai import OpenAI
from huggingface_hub import snapshot_download

model_id = "meta-llama/Meta-Llama-3-8B"
llm = LLM(model=model_id, enable_lora=True, max_lora_rank=16)

sampling_params_oasst = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=500)
oasst_lora_id = "kaitchup/Meta-Llama-3-8B-oasst-Adapter"
oasst_lora_path = snapshot_download(repo_id=oasst_lora_id)
oasstLR = LoRARequest("oasst", 1, oasst_lora_path)

sampling_params_xlam = SamplingParams(temperature=0.0, max_tokens=500)
xlam_lora_id = "kaitchup/Meta-Llama-3-8B-xLAM-Adapter"
xlam_lora_path = snapshot_download(repo_id=xlam_lora_id)
xlamLR = LoRARequest("xlam", 2, xlam_lora_path)

#### lora_request=oasstLR
prompts_oasst = [
    "### Human: Check if the numbers 8 and 1233 are powers of two.### Assistant:",
    "### Human: What is the division result of 75 divided by 1555?### Assistant:",
]
outputs = llm.generate(prompts_oasst, sampling_params_oasst, lora_request=oasstLR)
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
    print('------')


#### lora_request=xlamLR
prompts_xlam = [
    "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>",
    "<user>What is the division result of 75 divided by 1555?</user>\n\n<tools>",
]

outputs = llm.generate(prompts_xlam, sampling_params_xlam, lora_request=xlamLR)
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
    print('------')


## Serving Multiple Adapters with vLLM
oasst_lora_id = "kaitchup/Meta-Llama-3-8B-oasst-Adapter"
oasst_lora_path = snapshot_download(repo_id=oasst_lora_id)
xlam_lora_id = "kaitchup/Meta-Llama-3-8B-xLAM-Adapter"
xlam_lora_path = snapshot_download(repo_id=xlam_lora_id)

# start the vLLM server
nohup vllm serve meta-llama/Meta-Llama-3-8B --enable-lora --lora-modules oasst={oasst_lora_path} xlam={xlam_lora_path} &

## with openai
model_id = "meta-llama/Meta-Llama-3-8B"
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
prompts = [
    "### Human: Check if the numbers 8 and 1233 are powers of two.### Assistant:",
    "### Human: What is the division result of 75 divided by 1555?### Assistant:",
]
completion = client.completions.create(model="oasst",
                                      prompt=prompts, temperature=0.7, top_p=0.9, max_tokens=500)
print("Completion result:", completion)

prompts = [
    "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>",
    "<user>What is the division result of 75 divided by 1555?</user>\n\n<tools>",
]
completion = client.completions.create(model="xlam",
                                      prompt=prompts, temperature=0.0, max_tokens=500)
print("Completion result:", completion)

