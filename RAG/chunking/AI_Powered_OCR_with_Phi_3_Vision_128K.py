### From https://ai.gopubby.com/ai-powered-ocr-with-phi-3-vision-128k-the-future-of-document-processing-7be80c46bd16

"""
flash_attn==2.5.8
numpy==1.24.4
Pillow==10.3.0
Requests==2.31.0
torch==2.3.0
torchvision==0.18.0
transformers==4.40.2
"""

from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

class Phi3VisionModel:
    def __init__(self, model_id="microsoft/Phi-3-vision-128k-instruct", device="cuda"):
        """
        Initialize the Phi3VisionModel with the specified model ID and device.
        
        Args:
            model_id (str): The identifier of the pre-trained model from Hugging Face's model hub.
            device (str): The device to load the model on ("cuda" for GPU or "cpu").
        """
        self.model_id = model_id
        self.device = device
        self.model = self.load_model()  # Load the model during initialization
        self.processor = self.load_processor()  # Load the processor during initialization
    
    def load_model(self):
        """
        Load the pre-trained language model with causal language modeling capabilities.
        
        Returns:
            model (AutoModelForCausalLM): The loaded model.
        """
        print("Loading model...")
        # Load the model with automatic device mapping and data type adjustment
        return AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="auto",  # Automatically map model to the appropriate device(s)
            torch_dtype="auto",  # Use an appropriate torch data type based on the device
            trust_remote_code=True,  # Allow execution of custom code for loading the model
            _attn_implementation='flash_attention_2'  # Use optimized attention implementation
        ).to(self.device)  # Move the model to the specified device
    
    def load_processor(self):
        """
        Load the processor associated with the model for processing inputs and outputs.
        
        Returns:
            processor (AutoProcessor): The loaded processor for handling text and images.
        """
        print("Loading processor...")
        # Load the processor with trust_remote_code=True to handle any custom processing logic
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
    
    def predict(self, image_url, prompt):
        """
        Perform a prediction using the model given an image and a prompt.
        
        Args:
            image_url (str): The URL of the image to be processed.
            prompt (str): The textual prompt that guides the model's generation.
        
        Returns:
            response (str): The generated response from the model.
        """
        # Load the image from the provided URL
        image = Image.open(requests.get(image_url, stream=True).raw)
        
        # Format the input prompt template for the model
        prompt_template = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Process the inputs, converting the prompt and image into tensor format
        inputs = self.processor(prompt_template, [image], return_tensors="pt").to(self.device)
        
        # Set generation arguments for the model's response generation
        generation_args = {
            "max_new_tokens": 500,  # Maximum number of tokens to generate
            "temperature": 0.7,     # Sampling temperature for diversity in generation
            "do_sample": False      # Disable sampling for deterministic output
        }
        print("Generating response...")
        # Generate the output IDs using the model, skipping the input tokens
        output_ids = self.model.generate(**inputs, **generation_args)
        output_ids = output_ids[:, inputs['input_ids'].shape[1]:]  # Ignore the input prompt in the output
        
        # Decode the generated output tokens to obtain the response text
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response

# Initialize the model
phi_model = Phi3VisionModel()

# Example prediction
image_url = "https://example.com/sample_image.png"  # URL of the sample image
prompt = "Extract the data in json format."  # Prompt for model guidance
response = phi_model.predict(image_url, prompt)  # Get the response from the model

print("Response:", response)  # Print the generated response



