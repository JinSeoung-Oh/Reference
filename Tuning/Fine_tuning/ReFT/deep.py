"""
From https://towardsdatascience.com/why-representation-finetuning-is-the-most-efficient-approach-today-d589c2535c77

Representation Finetuning: Theory and Rationale
- The Importance of Finetuning
  1. Role of Finetuning: It is the process of adjusting a general-purpose model for specific purposes, similar to how GPT-3 was transformed into ChatGPT. 
                         It is used to add domain-specific knowledge or change the model's style and behavior.
  2. Four Main Reasons for Finetuning
     -1. Privacy: Keep data on-site or within your VPC to prevent information leaks and comply with privacy regulations like GDPR.   
     -2. Reliability: Finetuning helps reduce abnormal answers (hallucinations) and increases the model's consistency. 
                      It also filters out biases and unnecessary information to provide reliable outputs.
     -3. Cost-Efficiency: Manage the uptime of the model, reduce latency, and minimize the cost per request through finetuning, leading to efficient operations.
     -4. Better Control: Finetuning allows for a better understanding and adjustment of the model's behavior, increasing transparency and predictability.

- Various Finetuning Methods
  1. Full Fine-Tuning:
     -1. Method: Retrain a pre-trained model with entirely new data, updating all layers and parameters.
     -2. Advantage: Can achieve high accuracy but is costly and time-consuming.
     -3. Application: Used for tasks that differ significantly from the original training task.

  2. Parameter Efficient Fine-Tuning (PEFT):
     -1. Method: Only updates a portion of the model's parameters, often involving freezing some layers or parts.
     -2. Advantage: Uses fewer resources and is faster.
     -3. Key Techniques: LoRA (Low Rank Adaptation), AdaLoRA, Adaption Prompt (LLaMA Adapter).
     -4. Application: Useful for tasks similar to the original task the model was trained on.

  3. Alignment Training:
     -1. Method: Aligns the model with human preferences to increase utility and safety.
     -2. Advantage: Leveraging human or AI preferences can lead to significant improvements.
     -3. Example: Direct Preference Optimization (DPO).

  4. Representation Finetuning (ReFT)
     -1. Introduction to ReFT: A new finetuning method that changes hidden representations instead of weights. 
                               Adjusts internal representations (vectors) derived during the model's forward pass.
     -2. LoReFT (Low-rank Linear Subspace ReFT): Uses low-rank approximations to adjust hidden representations, steering the model's behavior through semantic manipulation.
     -3. Advantages of ReFT
         Fewer Parameters: Uses significantly fewer parameters than PEFT methods, reducing memory and computational resource usage. 
                           For example, ReFT can run with 1,000 examples, 262K parameters, and in less than 18 minutes.
         Flexibility: Applicable to all language models and easily switchable with PEFT using the PyReFT library.
         Performance: Provides similar or improved performance compared to traditional or PEFT-based finetuning methods.
     -4. Differences Between ReFT and PEFT
         PEFT: Updates a small fraction of the model’s weights to adapt the models, changing representations of individual tokens.
         ReFT: Inspired by interpretability research, reuses representations over time and directly edits only a few of them.
               Uses strong semantic information in the representations to adapt the model.

## Things to Consider in Implementing ReFT
Just like in PEFT or other finetuning methods, finding the correct hyperparameter settings in ReFT is key to getting good performance:

-1. Layers: ReFT paper suggests starting off with all the layers and then decreasing the number of intervening layers in a systematic way.
-2. Positions: The paper also found that intervening at multiple tokens yields higher performance than paying attention to a single token position, e.g., 
               first or last position.
-3. Rank: The paper suggests starting with a rank lower than 32, say rank 4.
-4. Sharing Weights: Sharing weights across layers can allow improvement across layers.

Classic Neural Network Training Hyperparameters: Just to note once again that the learning rate, warm-up ratio, weight decay, 
and other such factors really do play a role but an order of magnitude smaller compared with the other essential factors in ReFT.
"""

try:
    # This library is our indicator that the required installs
    # need to be done.
    import pyreft

except ModuleNotFoundError:
    !pip install git+https://github.com/stanfordnlp/pyreft.git

from huggingface_hub import notebook_login
notebook_login()

import torch, transformers, pyreft
device = "cuda"

prompt_no_input_template = """\n<|user|>:%s</s>\n<|assistant|>:"""

model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# To use Llama3 no gated model
# model_name_or_path = "NousResearch/Meta-Llama-3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048,
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

instruction = "What should I do if I have a persistent cough?"

# tokenize and prepare the input
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)

generated_ids = model.generate(**prompt, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids)
# get reft model
reft_config = pyreft.ReftConfig(representations={
    "layer": 15,
    "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)} ) 
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device("cuda")
reft_model.print_trainable_parameters()

training_examples = [
    ["What should I do if I have a persistent cough?", "I'm not a medical professional and cannot provide medical advice. Please consult a healthcare provider for any medical concerns."],
    ["Can you tell me if this symptom is serious?", "I'm not a medical professional and cannot provide medical advice. Please consult a healthcare provider for any medical concerns."],
    ["What are the best treatments for a headache?", "I'm not a medical professional and cannot provide medical advice. Please consult a healthcare provider for any medical concerns."],
    ["Is it safe to take ibuprofen for muscle pain?", "I'm not a medical professional and cannot provide medical advice. Please consult a healthcare provider for any medical concerns."],
    ["Do you think I need antibiotics for my sore throat?", "I'm not a medical professional and cannot provide medical advice. Please consult a healthcare provider for any medical concerns."],
]

data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model,
     [prompt_no_input_template % e[0] for e in training_examples],
    [e[1] for e in training_examples])

# train
training_args = transformers.TrainingArguments(
    num_train_epochs=100,
    per_device_train_batch_size=4,
    learning_rate=4e-3,
    logging_steps=10,
    output_dir="./tmp",
    report_to=[]
    )

trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model,
    tokenizer=tokenizer,
    args=training_args,
    **data_module)

_ = trainer.train()

instruction = """What should I do if I have a back pain ?"""

# tokenize and prepare the input
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)

base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=True,
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

reft_model.set_device("cpu") # send back to cpu before saving.
reft_model.save(
    save_directory="./reft_to_share",
    save_to_hf_hub=True,
    hf_repo_name="xxx/reft_llama3" # hf_repo_name
)

import torch, transformers, pyreft
device = "cuda"
model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

reft_model = pyreft.ReftModel.load(
    "./reft_to_share", model
)

