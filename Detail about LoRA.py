# from https://towardsdatascience.com/lora-intuitively-and-exhaustively-explained-e944a6bff46b

### LoRA
# “Low-Rank Adaptation” (LoRA) is a form of “parameter efficient fine tuning” (PEFT), 
# which allows one to fine tune a large model using a small number of learnable parameters
#  1. We can think of fine tuning as learning changes to parameters, instead of adjusting parameters themselves
#  2. We can try to compress those changes into a smaller representation by removing duplicate information.
#  3. We can “load” our changes by simply adding them to the pre-trained parameters

## 1) Fine Tuning as Parameter Changes
#   The most basic approach to fine tuning consists of iteratively updating parameters
#   Input data --> LLM parameters(Generate predecition) --> LLM Gradients(Calculate Gradients) --> Loss Function
#   --> use loss to update Model --> LLM Gradients(Calculate Required Changes from gradients) --> LLM parameter(update LLM para)

#   LoRA --> fine tuning as learning parameter changes
#   Freeze the model parameters, exactly how they are, 
#   and learn the changes to those parameters necessary to make the model perform better at the fine tuned task.

## 2) Parameter Change Compression
#   From the perspective of LoRA, understanding that weights are actually a matrix is incredibly important, 
#   as a matrix has certain properties which we can be leveraged to condense information.
 
## The Core Idea Behind LoRA
#   LoRA thinks of tuning not as adjusting parameters, but as learning parameter changes. 
#   With LoRA we don’t learn the parameter changes directly,
#   however; we learn the factors of the parameter change matrix
#   This idea of learning factors of the change matrix relies on the core assumption 
#   that weight matrices within a large language model have a lot of linear dependence, 
#   as a result of having significantly more parameters than is theoretically required. 
#   Over parameterization has been shown to be beneficial in pre-training 
#   (which is why modern machine learning models are so large). 
#   The idea behind LoRA is that, once you’ve learned the general task with pre-training, 
#   you can do fine tuning with significantly less information.

## LoRA Rank
# LoRA has a hyperparameter, named r, which describes the depth of the A and B matrix used to construct
# the change matrix discussed previously. Higher r values mean larger A and B matrices, 
# which means they can encode more linearly independent information in the change matrix
# Generally, in selecting r, the advice I’ve heard is the following: 
#   When the data is similar to the data used in pre-training, a low r value is probably sufficient. 
#   When fine tuning on very new tasks, which might require substantial logical changes within the model,
#   a high r value may be required.

# Python Example
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

#loading model
model = AutoModelForCausalLM.from_pretrained(
    # "bigscience/bloom-3b",
    # "bigscience/bloom-1b1",
    "bigscience/bloom-560m",
    torch_dtype=torch.float16,
    device_map='auto',
)

#loading tokenizer for this model (which turns text into an input for the model)
tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")

from peft import LoraConfig, get_peft_model

#defining how LoRA will work in this particular example
config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#this actually overwrites the model in memory, so
#the rename is only for ledgibility.
peft_model = get_peft_model(model, config)

trainable_params = 0
all_param = 0

#iterating over all parameters
for _, param in peft_model.named_parameters():
    #adding parameters to total
    all_param += param.numel()
    #adding parameters to trainable if they require a graident
    if param.requires_grad:
        trainable_params += param.numel()

#printing results
print(f"trainable params: {trainable_params}")
print(f"all params: {all_param}")
print(f"trainable: {100 * trainable_params / all_param:.2f}%")

from datasets import load_dataset
qa_dataset = load_dataset("squad_v2")

def create_prompt(context, question, answer):
  if len(answer["text"]) < 1:
    answer = "Cannot Find Answer"
  else:
    answer = answer["text"][0]
  prompt_template = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}</s>"
  return prompt_template

#applying the reformatting function to the entire dataset
mapped_qa_dataset = qa_dataset.map(lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))

import transformers

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=mapped_qa_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=100,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=1,
        output_dir='outputs',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model_id = "BLOOM-560m-LoRA"
peft_model.save_pretrained(model_id)


## Testing example
from IPython.display import display, Markdown

def make_inference(context, question):

    #turn the input into tokens
    batch = tokenizer(f"**CONTEXT:**\n{context}\n\n**QUESTION:**\n{question}\n\n**ANSWER:**\n", return_tensors='pt', return_token_type_ids=False)
    #move the tokens onto the GPU, for inference
    batch = batch.to(device='cuda')

    #make an inference with both the fine tuned model and the raw model
    with torch.cuda.amp.autocast():
        #I think inference time would be faster if these were applied,
        #but the fact that LoRA is not applied allows me to experiment
        #with before and after fine tuning simultaniously

        #raw model
        peft_model.disable_adapter_layers()
        output_tokens_raw = model.generate(**batch, max_new_tokens=200)

        #LoRA model
        peft_model.enable_adapter_layers()
        output_tokens_qa = peft_model.generate(**batch, max_new_tokens=200)

    #display results
    display(Markdown("# Raw Model\n"))
    display(Markdown((tokenizer.decode(output_tokens_raw[0], skip_special_tokens=True))))
    display(Markdown("\n# QA Model\n"))
    display(Markdown((tokenizer.decode(output_tokens_qa[0], skip_special_tokens=True))))
