### From https://medium.com/@bernardomotab/ocr-llm-or-multimodal-models-paligemma2-fine-tuning-with-qlora-on-a-custom-dataset-9f50347cac3d

!pip install -q -U datasets bitsandbytes peft git+https://github.com/huggingface/transformers.git

from datasets import load_dataset, concatenate_datasets
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, PaliGemmaProcessor, AutoProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
import os

!huggingface-cli login --token $HF_TOKEN --add-to-git-credential

device = "cuda"  # Use GPU for training
model_id = "google/paligemma2-3b-pt-224"  # Pre-trained model identifier
dataset_path = "bernardomota/establishment-name-vqa"  # Custom dataset path
model_output = "paligemma2-qlora-st-vqa-estnamevqa-224"  # Output directory for the trained model

def resize_and_process(batch):
    """
    Resize images in the batch if necessary and return the updated batch.
    Args:
        batch (dict): A dictionary containing images and potentially other data.
    Returns:
        dict: The updated batch with resized images.
    """
    max_size = 640
    images = batch['image']
    # Resize each image in the batch
    resized_images = []
    for img in images:
        width, height = img.size
        if max(width, height) > max_size:
            resize_ratio = max_size / max(width, height)
            new_width = int(width * resize_ratio)
            new_height = int(height * resize_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_images.append(img)
    batch['image'] = resized_images

    return batch

ds_custom = load_dataset(dataset_path, trust_remote_code=True)
train_ds_custom = ds_custom["train"]
val_ds_custom = ds_custom["validation"]
train_ds_custom = train_ds_custom.map(resize_and_process, batched=True)
val_ds_custom = val_ds_custom.map(resize_and_process, batched=True)
print(train_ds_custom)
print(val_ds_custom)


# Function to process 'qas' and return expanded rows
def process_qas(examples):
    # Flatten the qas list and extract questions, answers, and images efficiently
    questions = [qa['question'] for qas_list in examples['qas'] for qa in qas_list]
    answers = [qa['answers'][-1] for qas_list in examples['qas'] for qa in qas_list]
    images = [image for image, qas_list in zip(examples['image'], examples['qas']) for _ in qas_list]

    return {'question': questions, 'image': images, 'answer': answers}

ds_stvqa = load_dataset('vikhyatk/st-vqa')['train']
ds_stvqa_sample = ds_stvqa.train_test_split(test_size=0.9)['train']
ds_stvqa_formatted = ds_stvqa_sample.map(process_qas, batched=True, remove_columns=['qas'])

# Split the dataset 90% for training, 10% for validation
ds_stvqa_formatted_split = ds_stvqa_formatted.train_test_split(test_size=0.1)

train_ds_stvqa = ds_stvqa_formatted_split['train']
val_ds_stvqa = ds_stvqa_formatted_split['test']
train_ds_stvqa = train_ds_stvqa.map(resize_and_process, batched=True)
val_ds_stvqa = val_ds_stvqa.map(resize_and_process, batched=True)
print(train_ds_stvqa)
print(val_ds_stvqa)

train_ds = concatenate_datasets([train_ds_custom, train_ds_stvqa])
val_ds = concatenate_datasets([val_ds_custom, val_ds_stvqa])
print(train_ds)
print(val_ds)

idx = -1
print(train_ds[idx])
train_ds[idx]['image']

processor = PaliGemmaProcessor.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map={"": 0}, 
    torch_dtype=torch.bfloat16
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

args = TrainingArguments(
    num_train_epochs=1,
    remove_unused_columns=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="paged_adamw_8bit",  # Optimizer choice
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    bf16=True,
    output_dir=model_output,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example['answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    
    # Process the inputs (questions and images) using the processor
    tokens = processor(
        text=texts, images=images, suffix=labels,
        return_tensors="pt", padding="longest",
        input_data_format="channels_last"
    )
    tokens = tokens.to(DTYPE).to(device)

    return tokens

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    args=args
)
trainer.train()

------------------------------------------------------------------------
from transformers.image_utils import load_image

model_id = "bernardomota/paligemma2-qlora-st-vqa-estnamevqa-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-pt-224")

url = "https://itajaishopping.com.br/wp-content/uploads/2023/02/burgerking-itajai-shopping.jpg"
image = load_image(url)


# Leaving the prompt blank for pre-trained models
prompt = "What is the name of the establishment?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
image
