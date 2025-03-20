### From https://pub.towardsai.net/googles-gemma-3-fine-tuning-made-simple-create-custom-ai-models-with-python-and-unsloth-fb937495e9db

# For installing Packages in Google Colab Notebook
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
!pip install --no-deps unsloth

# Install latest Hugging Face for Gemma-3!
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048,
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # More accurate, uses 2x memory
    full_finetuning = False, # To perform full finetuning
    # token = "<YOUR_HF_TOKEN>", # if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset = standardize_data_formats(dataset)

def act(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }
pass
dataset = dataset.map(act, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, 
    args = SFTConfig(
        dataset_text_field = "text", # The field of the dataset that is structured, and will be used for training
        per_device_train_batch_size = 2, # Number of samples processed per device in each batch
        gradient_accumulation_steps = 4, # Number of steps to accumulate gradients before performing a backward pass
        warmup_steps = 5, # The number of steps for gradual learning rate increase at the start of training
        # num_train_epochs = 1, # Set this for 1 full training run
        max_steps = 30, # The total number of training steps to perform
        learning_rate = 2e-4, # The learning rate for updating weights during training
        logging_steps = 1, # Frequency (in steps) for logging training metrics.
        optim = "adamw_8bit", # The optimizer
        weight_decay = 0.01, # The regularization technique to prevent overfitting
        lr_scheduler_type = "linear", # To control learning rate decay
        seed = 3407, 
        report_to = "none", # Platform for logging metrics can also be 'wandb'
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()


---------------------------------------------
### Inferencing the Fine-Tuned Model

from unsloth.chat_templates import get_chat_template

# Initializing the tokenizer with chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# Defining the prompt of the user
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]

# Applying the chat template on the prompt of the user
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

# Generating the output for the prompt from the model
outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!

    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)

# Decoding the generating output to text
tokenizer.batch_decode(outputs)

-----------------------------------------------
## Save
model.save_pretrained_merged("gemma-3-4b-Maxime-Labonne-FineTuned", tokenizer)

## Push huggingface
model.push_to_hub("<Your_HF_Account>/<Model_Name>", token = "YOUR_HF_TOKEN") 
tokenizer.push_to_hub("<Your_HF_Account>/<Model_Name>", token = "YOUR_HF_TOKEN")
