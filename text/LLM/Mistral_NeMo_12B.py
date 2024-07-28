## setup 
python -m venv mistral_nemo_env
source mistral_nemo_env/bin/activate  # On Windows, use: mistral_nemo_env\Scripts\activate
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate
!pip install git+https://github.com/huggingface/transformers.git

## download model
from huggingface_hub import snapshot_download
from pathlib import Path

model_path = Path.home().joinpath('mistral_models', 'Nemo-Instruct')
model_path.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
    local_dir=model_path
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = Path.home().joinpath('mistral_models', 'Nemo-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def generate_response(prompt, max_length=256, temperature=0.3):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        num_return_sequences=1
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Explain the concept of quantum entanglement in simple terms."
response = generate_response(prompt)
print(response)

### Fine-tuning for Specific Tasks
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("....")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# Start fine-tuning
trainer.train()

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

### Handling Long Contexts
tokenizer.model_max_length = 128000
model.config.max_position_embeddings = 128000

# Example of using a long context
long_prompt = "Your very long text here..." * 1000  # Repeat to create a long context
response = generate_response(long_prompt, max_length=2048)
print(response)

## set multi-gpu
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)







def chat_with_model():
    print("Chat with Mistral NeMo 12B (type 'exit' to end the conversation)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = generate_response(user_input)
        print("Mistral NeMo 12B:", response)

chat_with_model()



