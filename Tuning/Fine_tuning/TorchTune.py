### From https://medium.com/chat-gpt-now-writes-all-my-articles/pytorch-is-making-fine-tuning-llms-easy-with-torchtune-code-examples-for-lora-and-qlora-included-8ad157d27e2e

"""
1. What is Torchtune?
   Torchtune is built with four core principles that make it both powerful and user-friendly:

   -a. Simplicity and Extensibility: With a native-PyTorch design, Torchtune is modular and easy to adapt for custom workflows.
   -b. Correctness: The library emphasizes correctness by offering well-tested components, ensuring that you can trust its outputs.
   -c. Stability: Built on PyTorch’s solid foundation, Torchtune “just works,” minimizing headaches during experimentation.
   -d. Democratizing LLM Fine-Tuning: Torchtune works out-of-the-box across different hardware setups, making it accessible for a broad range of users.

2. Key Features of Torchtune:
   -a. Modular PyTorch Implementations: Supports popular LLM architectures.
   -b. Interoperability: Works seamlessly with Hugging Face Datasets and EleutherAI’s Eval Harness.
   -c. Distributed Training: Supports efficient large-scale training using features like Fully Sharded Data Parallel (FSDP2).
   -d. YAML Configs: Simplifies configuration for training runs.

3. Key Concepts in Torchtune
   Torchtune introduces two main concepts that make fine-tuning LLMs more accessible: Configs and Recipes.

   -a. Configs
       Configurations in Torchtune are handled via YAML files, allowing you to set up training parameters 
       (datasets, model architecture, batch size, learning rate, etc.) without changing the underlying code. 
       This separation of configuration from code means you can experiment with different setups quickly and easily.

   -b. Recipes
       Recipes are predefined pipelines for fine-tuning and evaluating LLMs. Each recipe outlines a specific training method and applies optimized 
       techniques like FSDP2, Activation Checkpointing, and Reduced Precision training. 
       These recipes are tailored to specific model families, such as Llama2 or Mistral, and make it easy to start fine-tuning with minimal 
       boilerplate.

4. Design Principles
   Torchtune embodies the PyTorch philosophy: usability above all else. Here’s how this design translates into practical benefits:

   -a. Native PyTorch: All core functionality is written in PyTorch, offering complete flexibility for experienced users.
   -b. Simplicity and Extensibility: The library avoids complex inheritance and abstractions, making it easy to read, understand, and extend.
   -c. Modular Components: Torchtune emphasizes reusable components over monolithic architectures, allowing users to mix and match functionality as needed.
   -d. Correctness: Each component and recipe undergoes extensive unit testing to ensure they align with reference implementations and benchmarks.
"""
#### Step 1: Download a Model
tune download meta-llama/Llama-2-7b-hf \
  --output-dir /tmp/Llama-2-7b-hf \
  --hf-token <ACCESS TOKEN>

#### Step 2: Selecting a Recipe
tune ls
$ tune run lora_finetune_single_device --config llama2/7B_lora_single_device

#### Step 3: Modifying the Config
# -1. Command-line overrides
$tune run <RECIPE> --config <CONFIG> epochs=1

# -2. Copy and edit
$ tune cp llama2/7B_lora_single_device custom_config.yaml

#### Step 4: Training the Model
$ tune run lora_finetune_single_device --config llama2/7B_lora_single_device epochs=1

######## CODE: Fine-Tuning Llama2 with LoRA using torchtune
import torch
from torchtune.models.llama2 import llama2_7b, lora_llama2_7b
from torchtune.modules.peft.peft_utils import get_adapter_params, set_trainable_params

# Ensure you have downloaded the necessary model weights and tokenizer
# Example paths (adjust for your environment):
llama2_weights_path = 'path/to/llama2/weights'
tokenizer_path = 'path/to/llama2/tokenizer'
# Load the base Llama2-7B model
base_model = llama2_7b(weights=llama2_weights_path, tokenizer=tokenizer_path)
# Print out a sample layer for inspection (optional)
print(base_model.layers[0].attn)

# Set up the Llama2 model with LoRA applied to specific layers
lora_model = lora_llama2_7b(
    lora_attn_modules=['q_proj', 'v_proj'],  # LoRA will be applied to these layers
    lora_rank=8,  # Low-rank decomposition
    lora_alpha=16  # Scaling factor for LoRA
)

# Load the base model weights into the LoRA model (setting strict=False)
lora_model.load_state_dict(base_model.state_dict(), strict=False)
# Ensure only LoRA parameters are trainable
lora_params = get_adapter_params(lora_model)
set_trainable_params(lora_model, lora_params)

# Print the number of total and trainable parameters
total_params = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"{total_params} total params, {trainable_params} trainable params, "
      f"{(100.0 * trainable_params / total_params):.2f}% of all params are trainable.")

# Finetune using torchtune's LoRA recipe
tune_command = """
tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config llama2/7B_lora \
lora_attn_modules=['q_proj', 'k_proj', 'v_proj', 'output_proj'] \
lora_rank=32 lora_alpha=64 output_dir=./lora_experiment_1
"""
# (You can modify the paths and parameters in the tune command as needed)

# Example of changing the LoRA rank and applying it to more layers
lora_experiment_model = lora_llama2_7b(
    lora_attn_modules=['q_proj', 'k_proj', 'v_proj', 'output_proj'],
    lora_rank=32,
    lora_alpha=64,
    apply_lora_to_mlp=True,  # Apply LoRA to MLP layers as well
    apply_lora_to_output=True  # Apply LoRA to output layers
)

# Running LoRA fine-tuning on a single GPU device:
tune run lora_finetune_single_device --config llama2/7B_lora_single_device

############### CODE: Fine-Tuning Llama2–7b with QLoRA Using Torchtune
import torch
from torchtune.models.llama2 import qlora_llama2_7b
from torchtune.trainers import LoRAFinetuneTrainer

# Initialize the QLoRA Llama2-7b model
qlora_model = qlora_llama2_7b(lora_attn_modules=["q_proj", "v_proj"])
# Prepare the trainer for fine-tuning with the QLoRA model
trainer = LoRAFinetuneTrainer(
    model=qlora_model,
    dataset_path="path_to_dataset",
    output_dir="output_dir",
    batch_size=2,
    max_steps=1000,
    logging_steps=100
)
# Start fine-tuning
trainer.train()
# Save the fine-tuned model
trainer.save_model("path_to_saved_model")







