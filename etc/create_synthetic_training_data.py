### From https://levelup.gitconnected.com/my-ai-pipeline-for-solving-the-no-training-data-problem-05038c788602
### From https://github.com/FareedKhan-dev/ai-vision-dataset-builder/?source=post_page-----05038c788602--------------------------------

git clone https://github.com/FareedKhan-dev/ai-vision-dataset-builder.git
cd ai-vision-dataset-builder

pip install -r requirements.txt

import os  # For interacting with the operating system
import math  # For mathematical operations
import io  # For file input and output operations
import ast  # For parsing and evaluating Python expressions
import base64  # For base64 encoding and decoding
from io import BytesIO  # For reading and writing files in memory

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting and visualizations

import cv2  # OpenCV library for computer vision tasks
from PIL import ImageDraw  # For image processing and drawing graphics

import torch  # PyTorch for deep learning
from diffusers import StableDiffusionPipeline  # For text-to-image generation with Stable Diffusion
from autodistill.detection import CaptionOntology  # For labeling/annotation tasks in object detection
from autodistill_grounding_dino import GroundingDINO  # For grounding and detection tasks
from openai import OpenAI  # OpenAI API for AI Chat

# Define the important objects that must be present in each generated prompt.
important_objects = "brown bear"  # For multiple objects, separate them with commas, e.g., "different kinds of bear, bottles, ... etc."

# Specify the number of prompts to generate.
number_of_prompts = 50  # Define the number of prompts to generate for the image generation task.

# Provide a brief description of the kind of images you want the prompts to depict.
description_of_prompt = "brown bear in different environments"  # Describe the scenario or context for the image generation.

# Generate a formatted instruction set to produce image generation prompts.
# This formatted string will help in creating detailed and diverse prompts for the computer vision model.

base_prompt = f'''
# List of Important Objects:
# The objects listed here must be included in every generated prompt.
Important Objects that must be present in each prompt:
{important_objects}

# Input Details:
# The task is to generate a specific number of prompts related to the description provided.
Input:
Generate {number_of_prompts} realistic prompts related to {description_of_prompt} for image generation.

# Instructions for Prompt Generation:
# - Each prompt should depict real-life behaviors and scenarios involving the objects.
# - All important objects should be included in every prompt.
# - Ensure that the objects are captured at varying distances from the camera:
#   - From very close-up shots to objects in the far background.
# - The prompts should be diverse and detailed to cover a wide range of use cases.

# Output Format:
# - The output should be a Python list containing all the generated prompts as strings.
# - Each prompt should be enclosed in quotation marks and separated by commas within the list.
Output:
Return a Python list containing these prompts as strings for later use in training a computer vision model.
[prompt1, prompt2, ...]
'''

# (Optional) - Print the formatted instruction set for generating prompts.
print(base_prompt)

####### 1. Using LLMs API (eg. OpenAI’s GPT)
# Initialize the OpenAI API client with your API key.
openai_chat = OpenAI(
    api_key="YOUR_OPENAI_API_KEY"
)

# Generate prompts for image generation using the OpenAI API.
completion = openai_chat.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": base_prompt+ "\n\n your response: [prompt1, prompt2, ...] and do not say anything else and i will be be using ast.literal_eval to convert the string to a list"}]
)

# Extract the generated prompts from the API response.
response = completion.choices[0].message.content

# Extract the part of the string that contains the variable definition
variable_definition = response.strip()

# Fix the formatting issue by ensuring the string is a valid Python list
if variable_definition.endswith(","):
 variable_definition = variable_definition[:-1] + "]"

# Use ast.literal_eval to safely evaluate the variable definition
prompts = ast.literal_eval(variable_definition)

# Print the first few prompts to verify the output
print(prompts[0:5])


# OUTPUT #
[ 
  'A brown bear in a dense forest, ...', 
  'A brown bear walking alone in a ...', 
  "A close-up shot of a brown bear's face ...", 
  ...
]

####### Image Generation
# Defining the model name and device to use for the Stable Diffusion pipeline.
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Load the Stable Diffusion pipeline with the specified model and device.
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Extract the first 5 prompts for generating images. (sample data)
sample_prompts = prompts[:10]

# Generate images based on the sample prompts using the Stable Diffusion pipeline.
images = pipe(sample_prompts).images

# Convert the generated images to a format that can be displayed in a DataFrame.
synthetic_data = pd.DataFrame({'Prompt': sample_prompts, 'Image': images})

# Display the synthetic data containing prompts and the corresponding generated images.
synthetic_data

# Define a function to display the images generated from the prompts.
def display_images(dataframe):
    
    # Extract the images from the DataFrame
    images = dataframe['Image']

    # Set up the grid
    max_images_per_row = 5
    rows = (len(images) + max_images_per_row - 1) // max_images_per_row
    columns = min(len(images), max_images_per_row)

    # Create a figure and axis
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 5))

    # Flatten axes if there's only one row
    if rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Display each image
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Show the grid of images
    plt.show()

# Display the synthetic images generated from the prompts.
display_images(synthetic_data)

######## Validating the Generated Images
# Define the prompt for the object detection task.
validation_prompt = "Analyze the provided image and determine if it depicts a real bear, which is an animal, excluding any other types of objects or representations. Respond strictly with 'True' for yes or 'False' for no."

# Initialize the OpenAI API client with your Nebius API key.
openai_chat = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key="YOUR_NEBIUS_API_KEY"
)

# Define a function to validate images using the OpenAI API.
def validate_images(validation_prompt, images):

    # Initialize an empty list to store the validation results
    bools = []

    # Function to encode a PIL image to base64
    def encode_image_pil(image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")  # Save the image to the buffer in JPEG format
        buffer.seek(0)  # Rewind the buffer to the beginning
        return base64.b64encode(buffer.read()).decode("utf-8")  # Convert to base64

    # Iterate through your images
    for image in images:
        # Convert the PIL image to base64
        base64_image = encode_image_pil(image)

        # Prepare the API payload
        response = openai_chat.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validation_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
        )

        # Append the response to the list
        bools.append(response.choices[0].message.content.replace('.', '').replace('\n', ''))

    # Convert the list of strings to a list of booleans
    bools = [ast.literal_eval(item) for item in bools]
    return bools

# Validate images and add the results as a new column in the dataframe
synthetic_data['bear_class'] = validate_images(validation_prompt, synthetic_data['Image'])

# Count rows where "Bear Class" is False
false_count = synthetic_data[synthetic_data["Bear Class"] == False].shape[0]

# Define a function to regenerate and validate images based on the validation results.
def regenerate_and_validate_images(dataframe, validation_prompt, pipe):

    # Get rows where bear_class is False (i.e., the image does not depict a bear) and runs the image generation process again for those rows
    rows_to_regenerate = dataframe[dataframe['bear_class'] == False]

    # Extract indices and prompts separately
    indices_to_regenerate = rows_to_regenerate.index
    prompts_to_regenerate = rows_to_regenerate['Prompt'].tolist()

    # Generate images based on the prompts that need to be regenerated.
    images_to_regenerate = pipe(prompts_to_regenerate).images

    # Iterate over the indices and the newly generated images
    for idx, img in zip(indices_to_regenerate, images_to_regenerate):
        dataframe.at[idx, 'Image'] = img

    # Validate only the rows that were regenerated
    dataframe.loc[indices_to_regenerate, 'bear_class'] = validate_images(validation_prompt, dataframe.loc[indices_to_regenerate, 'Image'])

    return dataframe

# Call the function to regenerate and validate images
synthetic_data = regenerate_and_validate_images(synthetic_data, validation_prompt, pipe)

######## Labeling Images
# Defining the CaptionOntology for the object "bear" in the generated images.
ontology=CaptionOntology(
    {
        "bear": "bear" # Define the ontology for the object "bear"
    }
)

# Initialize the GroundingDINO model with the defined ontology.
base_model = GroundingDINO(ontology=ontology)

# Create a temporary directory for saving images
temp_dir = "temp_images"
os.makedirs(temp_dir, exist_ok=True)

# Save the images to the temporary directory
for idx, img in enumerate(synthetic_data['Image']):
    file_path = os.path.join(temp_dir, f"image_{idx}.jpg")
    img.save(file_path)  # Save the PIL image

# Label the images using the GroundingDINO model
base_model.label(temp_dir,  # Pass the list of image file paths
                 extension=".jpg",
                 output_folder="labeled_images")

# Optional: Clean up the temporary directory after labeling (if desired)
import shutil
shutil.rmtree(temp_dir)

# Paths to the train and valid directories
train_labels_dir = "labeled_images/train/labels"
valid_labels_dir = "labeled_images/valid/labels"

# Function to map labels to dataframe
def map_labels_to_dataframe(df, train_labels_dir, valid_labels_dir):
    # Combine all label paths
    label_paths = {}
    for label_dir in [train_labels_dir, valid_labels_dir]:
        if os.path.exists(label_dir):
            label_files = sorted(os.listdir(label_dir))  # Ensure order matches `image_{index}`
            for label_file in label_files:
                if label_file.endswith(".txt"):
                    label_index = int(label_file.split('_')[1].split('.')[0])  # Extract index from filename
                    label_paths[label_index] = os.path.join(label_dir, label_file)

    # Read labels and map to dataframe
    labels = []
    for idx in range(len(df)):
        label_path = label_paths.get(idx, None)
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_content = f.read().strip()  # Read the bounding box info
                labels.append(label_content)
        else:
            labels.append("")  # No label found for this index

    # Assign labels to dataframe
    df['Labels'] = labels
    return df

# Map labels to the dataframe
synthetic_data = map_labels_to_dataframe(synthetic_data, train_labels_dir, valid_labels_dir)

# Optional: Clean up the temporary directory after labeling (if desired)
import shutil
shutil.rmtree("labeled_images")
print(false_count)





