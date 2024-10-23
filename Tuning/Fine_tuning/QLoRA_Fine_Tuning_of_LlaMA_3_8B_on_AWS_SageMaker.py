### From https://medium.com/@MUmarAmanat/qlora-fine-tuning-of-llama-3-8b-on-aws-sagemaker-2a6e787d726b

%pip install s3fs
%pip install datasets

import sagemaker
import pandas as pd
import boto3
import io
import os
import pickle

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

job_df = pd.read_csv(f"s3://{os.path.join(BUCKET_NAME, DATA_DIR, CSV_FILE)}").drop(columns=["Unnamed: 0"])
job_df['Target_cat'] = job_df['Job Title'].astype('category')
job_df['Target'] = job_df['Target_cat'].cat.codes

## category map
with open(r"./job_category.pickle", "wb") as output_file:
    pickle.dump(category_map, output_file)

# Function to split data for each category
def split_data(group):
    train, temp = train_test_split(group, test_size=0.2, random_state=42)  # 80% train, 20% temp
    val, test = train_test_split(temp, test_size=0.5, random_state=42)     # 50% of temp -> 10% val, 10% test
    return train, val, test


# Initialize empty dataframes to store results
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

# Apply split for each category
for _, group in job_df.groupby('Target_cat'):
    train, val, test = split_data(group)
    train_df = pd.concat([train_df, train])
    val_df = pd.concat([val_df, val])
    test_df = pd.concat([test_df, test])

# Display the size of each split
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

### Converting Pandas DataFrame to Hugging Face Datasets
def return_hf_dataset(train_df, val_df, test_df):
    df_train = train_df.copy()
    df_val = val_df.copy()
    df_test = test_df.copy()
    print("[INFO] Train, Test, and Val set shape", df_train.shape, df_test.shape, df_val.shape)
    
    dataset_train = Dataset.from_pandas(df_train.drop('Target_cat', axis=1).reset_index())
    dataset_val = Dataset.from_pandas(df_val.drop('Target_cat', axis=1).reset_index())
    dataset_test = Dataset.from_pandas(df_test.drop('Target_cat', axis=1).reset_index())
    
    # Combine them into a single DatasetDict                                                              
    dataset = DatasetDict({
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test
    })
    return dataset
  
dataset.save_to_disk(f"s3://{BUCKET_NAME}/dataset/")

### Prepare Train Scripts
/opt/conda/bin/python train.py --epochs 1 --model_name <MODEL-NAME> --train_batch_size <BATCH-SIZE>

"""
-- /
    -- scripts/
              -- train.py
              -- requirements.txt
    -- 1_DatasetCreation_v2.ipynb
    -- 2_HFTrainer_v2.ipynb
"""
--------------------------------------------------------------------------------------------
import logging ## For logging info, warn, etc.
import sys ## For specifying stdout handler
import argparse ## Parse argument supply at execution time
import torch 
import pandas as pd
import os
import torch.nn.functional as F ## Functional API for configuring activation, loss, etc
import numpy as np

from datasets import load_from_disk, Dataset, DatasetDict ## HugginFace Datasets API
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)

from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

## Utility function for setting up PEFT fine-tuning method
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
## trasnformer API for selecting model, tokenizer, etc
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
## This improt is for accessing gated models
from huggingface_hub.hf_api import HfFolder

class CustomTrainer(Trainer):
  def __init__(self, *args, class_weights=None, **kwargs):
      super().__init__(*args, **kwargs)
      # Ensure label_weights is a tensor
      if class_weights is not None:
          self.class_weights = torch.tesnor(class_weights, dtype=torch.float32).to(self.args.device)
      else:
          self.class_weights = None

  def compute_loss(self, model, inputs, return_outputs=False):
      # Extract labels and convert them to long type for cross_entropy
      labels = inputs.pop("labels").long()

      # Forward pass
      outputs = model(**inputs)

      # Extract logits assuming they are directly outputted by the model
      logits = outputs.get('logits')

      # Compute custom loss with class weights for imbalanced data handling
      if self.class_weights is not None:
          loss = F.cross_entropy(logits, labels, weight=self.class_weights)
      else:
          loss = F.cross_entropy(logits, labels)

      return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),
            'accuracy':accuracy_score(predictions,labels)}
  
HfFolder.save_token('')  ## Make sure to use your own hf_api_token
-----------------------------------------------------------------------------------------------------------
##### Configuring QLoRA
lora_config = LoraConfig(r=32, #low-rank adaptation matrix rank 32,
                         lora_alpha=32, ## LoRA 
                         target_modules= ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  ## The modules(for example, attention blocks) to apply the LoRA update matrices.
                         lora_dropout = 0.05,
                         bias='none',
                         task_type='SEQ_CLS'
    )
-----------------------------------------------------------------------------------------------------------
##### Quantization Configuration
quantization_config = BitsAndBytesConfig(
                                      load_in_4bit = True, # enable 4-bit quantization
                                      bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
                                      bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
                                      bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
                          )
-----------------------------------------------------------------------------------------------------------
 model_name = args.model_name
  lora_config = LoraConfig(r=32, #rank 32,
                       lora_alpha=32, ## LoRA Scaling factor 
                       target_modules= ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  ## The modules(for example, attention blocks) to apply the LoRA update matrices.
                       lora_dropout = 0.05,
                       bias='none',
                       task_type='SEQ_CLS'
  )

  quantization_config = BitsAndBytesConfig(
                                      load_in_4bit = True, # enable 4-bit quantization
                                      bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
                                      bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
                                      bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
                          )

  model = AutoModelForSequenceClassification.from_pretrained(
                                                              model_name,
                                                              quantization_config=quantization_config,
                                                              num_labels=int(args.num_label)
                                                          )

     
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)
  logger.info(f"\nModel architecture: {model}")

  def print_number_of_trainable_model_parameters(model):
      trainable_model_params = 0
      all_model_params = 0
      for _, param in model.named_parameters():
          all_model_params += param.numel()
          if param.requires_grad:
              trainable_model_params += param.numel()
      return f'trainable model parameters: {trainable_model_params}\n \
              all model parameters: {all_model_params} \n \
              percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %'
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

MAX_LEN = 512
col_to_delete = ['index', 'Job Title']

def llama_preprocessing_function(examples):
    return tokenizer(examples['Job Description'], truncation=True, max_length=MAX_LEN)


dataset = load_from_disk(args.dataset_dir)
tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
tokenized_datasets = tokenized_datasets.rename_column("Target", "label")
tokenized_datasets.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

 training_args = TrainingArguments(
                                    output_dir = args.model_dir,
                                    learning_rate = 1e-4,
                                    per_device_train_batch_size = 8,
                                    per_device_eval_batch_size = 8,
                                    num_train_epochs = int(args.epochs),
                                    weight_decay = 0.01,
                                    evaluation_strategy = 'epoch',
                                    save_strategy = 'epoch',
                                    load_best_model_at_end = True,
                                    logging_dir=f"{args.output_data_dir}/logs",
                                    )
    
    trainer = CustomTrainer(
                            model = model,
                            args = training_args,
                            train_dataset = tokenized_datasets['train'],
                            eval_dataset = tokenized_datasets['val'],
                            tokenizer = tokenizer,
                            data_collator = collate_fn,
                            compute_metrics = compute_metrics,
                        )

    # train model
    trainer.train()

    # Saves the model to s3
    trainer.save_model(args.model_dir)
-------------------------------------------------------------------------------------------------------------
############# Requirements.txt
accelerate
bitsandbytes
s3fs
transformers
peft
datasets
-------------------------------------------------------------------------------------------------------------
!python ./scripts/train.py --epochs 1 --model_dir ./output/model   --model_name meta-llama/Meta-Llama-3-8B --num_label 15 --train_batch_size 32 --dataset_dir s3://<S3-PATH>/dataset/

-------------------------------------------------------------------------------------------------------------
############# Start training using SageMaker API
import sagemaker

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sagemaker.huggingface import HuggingFace

NUM_LABELS = 15

role = sagemaker.get_execution_role()

# hyperparameters, which are passed into the training job
hyperparameters = {'epochs': 1,
                     'train_batch_size': 32,
                     'model_name':'meta-llama/Meta-Llama-3-8B',
                     'num_label': NUM_LABELS,
                     'dataset_dir': 's3://<S3-PATH>/dataset/'
                 }

huggingface_estimator = HuggingFace(entry_point='train.py',
                                    source_dir='./scripts',
                                    instance_type='ml.p4d.24xlarge',
                                    instance_count=1,
                                    role=role,
                                    transformers_version='4.36',
                                    pytorch_version='2.1',
                                    py_version='py310',
                                    hyperparameters = hyperparameters
                                   )

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit()


