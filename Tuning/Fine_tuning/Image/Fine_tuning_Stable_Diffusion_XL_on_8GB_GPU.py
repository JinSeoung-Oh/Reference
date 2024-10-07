## From https://generativeai.pub/fine-tuning-stable-diffusion-xl-on-8gb-gpu-2324f4ec4352

"""
conda create -n sdxl python=3.10
conda activate sdxl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install diffusers transformers accelerate bitsandbytes xformers gradio
"""
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from accelerate import Accelerator
from diffusers import DDPMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance

def create_optimized_pipeline():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Load pipeline with optimizations
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Enable memory optimizations
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    
    return pipeline
  
def setup_unet_for_training():
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0/unet",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    return unet

def apply_memory_optimizations(model):
    # Enable gradient checkpointing
    model.enable_gradient_checkpointing()
    
    # Use 8-bit Adam optimizer
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    return optimizer

# Memory monitoring function
def log_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

class SDXLDataset(Dataset):
    def __init__(self, image_dir, metadata_file):
        self.image_dir = image_dir
        self.metadata = self.load_metadata(metadata_file)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def load_metadata(self, metadata_file):
        with open(metadata_file, 'r') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = f"{self.image_dir}/{item['file_name']}"
        image = Image.open(image_path).convert('RGB')
        
        return {
            'image': self.transform(image),
            'prompt': item['prompt']
        }    def load_metadata(self, metadata_file):
        with open(metadata_file, 'r') as f:
            return [json.loads(line) for line in f]

def setup_training():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16",
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )
    
    return accelerator, noise_scheduler

class TrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-5
        self.num_epochs = 100
        self.batch_size = 1
        self.gradient_accumulation_steps = 4
        self.mixed_precision = "fp16"

def train(config, model, dataset, accelerator, noise_scheduler):
    optimizer = apply_memory_optimizations(model)
    
    # Prepare for training
    model, optimizer, dataset = accelerator.prepare(
        model, optimizer, dataset
    )
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        for batch in dataset:
            with accelerator.accumulate(model):
                # Zero out gradients
                optimizer.zero_grad()
                
                # Get images and prompts
                images = batch['image']
                prompts = batch['prompt']
                
                # Generate noise
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch['image'].shape[0],), device=images.device
                )
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                
                # Predict noise
                noise_pred = model(noisy_images, timesteps, prompts).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                
        # Log progress
        accelerator.print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            accelerator.save_state(f"checkpoint-{epoch}")

####### Inference and Testing
def setup_inference_pipeline(trained_model_path):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=UNet2DConditionModel.from_pretrained(
            trained_model_path,
            torch_dtype=torch.float16
        ),
        torch_dtype=torch.float16,
    )
    pipeline.to("cuda")
    return pipeline

def generate_images(pipeline, prompt, num_images=4):
    images = pipeline(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=30,
    ).images
    return images

def calculate_fid(real_images, generated_images):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute()

########### Style Fine-tuning
def style_finetuning_example():
    # Configuration for style training
    style_config = TrainingConfig()
    style_config.learning_rate = 5e-6
    style_config.num_epochs = 50
    
    # Example dataset setup
    style_dataset = SDXLDataset(
        image_dir="path/to/style/images",
        metadata_file="path/to/style/metadata.jsonl"
    )
    
    # Training
    accelerator, noise_scheduler = setup_training()
    model = setup_unet_for_training()
    train(style_config, model, style_dataset, accelerator, noise_scheduler)
    
    return model

# Usage example
style_model = style_finetuning_example()
pipeline = setup_inference_pipeline("path/to/saved/style/model")
images = generate_images(pipeline, "A sunset in the style of Van Gogh")

########### Concept Fine-tuning
def concept_finetuning_example(concept_name):
    # Configuration for concept training
    concept_config = TrainingConfig()
    concept_config.learning_rate = 1e-6
    concept_config.num_epochs = 100
    
    # Example dataset setup
    concept_dataset = SDXLDataset(
        image_dir=f"path/to/{concept_name}/images",
        metadata_file=f"path/to/{concept_name}/metadata.jsonl"
    )
    
    # Training
    accelerator, noise_scheduler = setup_training()
    model = setup_unet_for_training()
    train(concept_config, model, concept_dataset, accelerator, noise_scheduler)
    
    return model

# Usage example
character_model = concept_finetuning_example("character_name")
pipeline = setup_inference_pipeline("path/to/saved/character/model")
images = generate_images(pipeline, "Character_name in a superhero costume")

########### Troubleshooting
# Out of Memory Errors
def handle_oom_errors():
    # Clear cache
    torch.cuda.empty_cache()
    
    # Move models to CPU temporarily
    def to_cpu(model):
        model.to("cpu")
        torch.cuda.empty_cache()
    
    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Use smaller image size
    config.image_size = 512  # Instead of 1024

#  Slow Training
def optimize_training_speed():
    # Use gradient accumulation
    config.gradient_accumulation_steps = 4
    
    # Enable torch compile
    model = torch.compile(model)
    
    # Use xformers attention
    model.enable_xformers_memory_efficient_attention()

# Monitoring and Debugging
class PerformanceMonitor:
    def __init__(self):
        self.training_steps = 0
        self.start_time = time.time()
    
    def log_step(self, loss):
        self.training_steps += 1
        if self.training_steps % 10 == 0:
            elapsed = time.time() - self.start_time
            steps_per_second = self.training_steps / elapsed
            print(f"Steps/second: {steps_per_second:.2f}, Loss: {loss:.4f}")

