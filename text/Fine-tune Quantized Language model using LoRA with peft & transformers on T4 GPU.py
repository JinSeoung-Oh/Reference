#from https://towardsdev.com/fine-tune-quantized-language-model-using-lora-with-peft-transformers-on-t4-gpu-287da2d5d7f1

## Setup environment
# The first step is to ensure we are using a T4 GPU on Google Colab
# And then 
# !pip install datasets transformers sentencepiece accelerate peft bitsandbytes evaluate

## Prepare dataset - In this, use Opus100 dataset
from datasets import get_dataset_config_names
configs = get_dataset_config_names("opus100")
print(configs)

from datasets import load_dataset
dataset = load_dataset("opus100", "en-fr")
dataset

## Data tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
  checkpoint, 
  load_in_8bit=True, 
  device_map="auto",
)

def preprocess_function(data):
  inputs = [ex['en'] for ex in data['translation']]
  targets = [ex['fr'] for ex in data['translation']]
  
  # tokeinze each row of inputs and outputs
  model_inputs = tokenizer(inputs, truncation=True)
  labels = tokenizer(targets, truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs
  
tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_dataset = tokenized_dataset['train'].shuffle(seed=42).select(range(2000))
val_dataset = tokenized_dataset['validation']

## Load the model
from transformers import AutoModelForSeq2SeqLM
model_id="t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

## Prepare model for training
from peft import PeftModel, prepare_model_for_int8_training, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

model = prepare_model_for_int8_training(model)
peft_config = LoraConfig(
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules=[ "k", "q", "v"],
)

peft_model = get_peft_model(model, peft_config)

## Training the model
from transformers import DataCollatorForSeq2Seq

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer
import torch

torch.set_default_dtype(torch.float32)

output_dir="en2fr"
num_epochs = 20
batch_size=8

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=num_epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
peft_model.config.use_cache = False # we can re-enable this for inference.

trainer.train()
peft_model.save_pretrained("translation/en2fr")
tokenizer.save_pretrained("translation/en2fr")

## Try out the model
peft_model.config.use_cache = True # silence the warnings. Please re-enable for inference!
context = tokenizer(["Do you want coffee?"], return_tensors='pt')
output = peft_model.generate(**context)
tokenizer.decode(output[0], skip_special_tokens=True)
