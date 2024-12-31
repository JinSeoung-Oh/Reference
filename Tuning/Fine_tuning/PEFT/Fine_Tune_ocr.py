### From https://bhavyajoshi809.medium.com/fine-tune-inference-of-idefics3-8b-on-custom-data-for-ocr-69e8bf61fecf

## Loading required libraries
import json
from tqdm import tqdm
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from datasets import load_dataset
from datasets import Dataset, DatasetDict, Image
import random



## Defining the fine-tuning type
## Choose whether to use LoRA or QLoRA fine-tuning
DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True
model_path = "HuggingFaceM4/Idefics3-8B-Llama3"


## Loading processor
processor = AutoProcessor.from_pretrained(
    model_path,
    do_image_splitting=False,
    local_files_only=True)

## Loading Model
if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())
else:
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to('cuda')
    
    # if you'd like to only fine-tune LLM
    for param in model.model.vision_model.parameters():
        param.requires_grad = False



## Loading Dataset
dataset_dict_path = 'path/to/idefics3-dataset.json'

with open(dataset_dict_path, 'r') as file:
    dataset_dict = json.load(file)


ds = load_dataset("json", data_files=dataset_dict_path)

# Spliting dataset into train and test
split_ds = ds['train'].train_test_split(test_size=0.1)
split_ds = split_ds.cast_column("image", Image())



## Creating Data Collector Class
class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        # Extract the image token ID from the processor's tokenizer
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
    
    def __call__(self, examples):
        texts = []
        images = []

        for example in examples:
            image = example["image"]
            question = example["question"]
            answer = example["answer"]

            # Prepare messages for user and assistant
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are given a image, your task is to extract out the desired text from the image as per the examples and classify the label of the image as 'chassis', 'vinplate', or 'other' based on the whole image."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            
            # Apply the chat template to format the messages
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())

            # Append image directly to the images list (no need to wrap in a list)
            images.append(image)

        # Process the texts and images into a batch
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Clone the input_ids from the batch and set padding/image tokens to -100
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100

        # Add the labels to the batch
        batch["labels"] = labels

        return batch
    
data_collator = MyDataCollator(processor)



## Defining Training Arguments
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    num_train_epochs=25,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=250,
    save_strategy="steps",
    save_steps=250*5,
    save_total_limit=1,
    optim="adamw_hf", # for 8-bit, pick paged_adamw_hf
    # evaluation_strategy="epoch",
    bf16=True,
    output_dir="path/to/idefics3_8b_qlora",
    remove_unused_columns=False
)


## Load trainer and train the model 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=split_ds['train'],
    eval_dataset=split_ds['test'],
)

trainer.train()
