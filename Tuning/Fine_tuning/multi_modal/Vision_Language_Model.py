## https://medium.com/google-developer-experts/ml-story-fine-tune-vision-language-model-on-custom-dataset-8e5f5dace7b1

!git clone https://github.com/NSTiwari/Fine-tune-IDEFICS-Vision-Language-Model
!pip install -q git+https://github.com/huggingface/transformers.gi 
!pip install -q accelerate datasets peft bitsandbytes

from datasets import Dataset, DatasetDict, Image
import pandas as pd
import os

# Define train and test size.
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 200
TEST_SIZE = 0.166 #

# Define the directory containing the images.
train_images_directory = '/content/Fine-tune-IDEFICS-Vision-Language-Model/dataset/images/train/'
test_images_directory = '/content/Fine-tune-IDEFICS-Vision-Language-Model/dataset/images/test/'

# Read the CSV Q&A text.
qa_text = pd.read_csv('/content/Fine-tune-IDEFICS-Vision-Language-Model/dataset/qa_text.csv')

# Create a list of image paths.
train_image_paths = [os.path.join(train_images_directory, f'train_{i}.jpg') for i in range(TRAIN_SAMPLES)]
test_image_paths = [os.path.join(test_images_directory, f'test_{i}.jpg') for i in range(TEST_SAMPLES)]
image_paths = train_image_paths + test_image_paths

# Create a list of other columns such as id, query, and answer.
ids = ids = qa_text['id'].tolist()
queries = qa_text['query'].tolist()
answers = qa_text['answers'].tolist()

# Create the dataset dictionary.
dataset_dict = {
    'id': ids,
    'image': image_paths,
    'query': queries,
    'answers': answers
}

# Create the dataset.
dataset = Dataset.from_dict(dataset_dict)

# Cast the 'image' column to Image type.
dataset = dataset.cast_column("image", Image())

# Split the dataset into train and test.
split_dataset = dataset.train_test_split(test_size=TEST_SIZE, shuffle=False)

# Push the dataset on Hugging Face Hub.
split_dataset.push_to_hub("NSTiwari/DocumentIDEFICS_QA")

from datasets import load_dataset

train_dataset = load_dataset("NSTiwari/DocumentIDEFICS_QA", split="train")
eval_dataset = load_dataset("NSTiwari/DocumentIDEFICS_QA", split="test")

##### Configure LoRA adapters
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA else None,
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
else:
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2", # Need GPUs like A100 or H100.
    ).to(DEVICE)

import random

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["query"]['en']
            answer = random.choice(example["answers"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
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
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)


#### Setup training parameters
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = "IDEFICS_DocVQA",
    learning_rate = 2e-4,
    fp16 = True,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 8,
    dataloader_pin_memory = False,
    save_total_limit = 3,
    evaluation_strategy ="steps",
    save_strategy = "steps",
    eval_steps = 10,
    save_steps = 25,
    max_steps = 25,
    logging_steps = 5,
    remove_unused_columns = False,
    push_to_hub=False,
    label_names = ["labels"],
    load_best_model_at_end = False,
    report_to = "none",
    optim = "paged_adamw_8bit",
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset
)

trainer.train()

## Evaluate the model
test_example = eval_dataset[0]
test_example["image"]

model.eval()

image = test_example["image"]
query = test_example["query"]


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Answer briefly."},
            {"type": "image"},
            {"type": "text", "text": query}
        ]
    }
]


text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
print(generated_texts)Question: What the location address of NSDA?
Answer: [‘1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036’, ‘1128 sixteenth st., N. W., washington, D. C. 20036’]


