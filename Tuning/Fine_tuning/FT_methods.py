## From https://medium.com/@yusufsevinir/13-%EF%B8%8F-fine-tuning-llms-for-precision-unlocking-the-full-potential-of-ai-57cebf802738

"""
Full Fine-Tuning on Custom Data

Pros: High accuracy, complete model control.
Cons: Expensive and computationally demanding; requires large datasets.
"""
import transformers
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset

# Load the model and tokenizer
model_name = "gpt2"  # Using GPT-2 as an example
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define your dataset
train_texts = ["Custom sentence 1.", "Custom sentence 2."]
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"]
})

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
---------------------------------------------------------------------------------------
"""
Parameter-Efficient Fine-Tuning (PEFT) 

1 LoRA (Low-Rank Adaptation)
Pros: Lower computational requirements, retains base model integrity.
Cons: May not achieve full accuracy compared to full fine-tuning.
"""
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# Load the base model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare your dataset
train_texts = ["Custom sentence 1.", "Custom sentence 2."]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"]
})

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
)

# Fine-tune with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
---------------------------------------------------------------------------------------
"""
Parameter-Efficient Fine-Tuning (PEFT) 

2 Prefix Tuning
Pros: Lightweight, minimal changes to the model.
Cons: Limited scope; best for modifying style and general behavior.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, PrefixTuningConfig, TaskType
from datasets import Dataset

# Load the base model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare your dataset
train_texts = ["Custom sentence 1.", "Custom sentence 2."]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"]
})

# Configure Prefix Tuning
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
)

# Apply Prefix Tuning to the model
model = get_peft_model(model, prefix_config)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
)

# Fine-tune with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
---------------------------------------------------------------------------------------
"""
Parameter-Efficient Fine-Tuning (PEFT) 

3 Adapters
Pros: Efficient, modular; multiple adapters can be used for different tasks.
Cons: Adds slight overhead; may not capture all task-specific nuances.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, AdaLoraConfig, TaskType
from datasets import Dataset

# Load the base model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare your dataset
train_texts = ["Custom sentence 1.", "Custom sentence 2."]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"]
})

# Configure Adapters
adapter_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_r=8,
    init_r=12,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
)

# Apply Adapters to the model
model = get_peft_model(model, adapter_config)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
)

# Fine-tune with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
---------------------------------------------------------------------------------------
"""
Parameter-Efficient Fine-Tuning (PEFT) 

4 Prompt Tuning 
Pros: Extremely lightweight, minimal computational resources.
Cons: May be less effective for complex tasks.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, PromptTuningConfig, TaskType
from datasets import Dataset

# Load the base model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare your dataset
train_texts = ["Custom sentence 1.", "Custom sentence 2."]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"]
})

# Configure Prompt Tuning
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
)

# Apply Prompt Tuning to the model
model = get_peft_model(model, prompt_config)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
)

# Fine-tune with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
---------------------------------------------------------------------------------------
"""
Optimizing Fine-Tuning for Performance
1 Batch Size and Learning Rate Adjustment
"""
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Larger batch size
    learning_rate=2e-5,             # Lower learning rate for fine-tuning
    num_train_epochs=2,
)
---------------------------------------------------------------------------------------
"""
Optimizing Fine-Tuning for Performance
2 Mixed Precision Training 
"""
training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,  # Enable mixed precision
)
---------------------------------------------------------------------------------------
"""
Optimizing Fine-Tuning for Performance
3 Distributed Training for Large Datasets 
"""
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    deepspeed="./deepspeed_config.json",  # Specify DeepSpeed configuration
)
---------------------------------------------------------------------------------------
"""
Evaluating Fine-Tuned Models
1 Accuracy Metrics  
"""
from datasets import load_metric

metric = load_metric("rouge")
predictions = ["Generated text..."]
references = ["Reference text..."]

results = metric.compute(predictions=predictions, references=references)
print(results)
---------------------------------------------------------------------------------------
"""
Evaluating Fine-Tuned Models
2 Qualitative Evaluation 
"""




