## https://medium.com/@datadrifters/got-ocr2-0-in-action-optical-character-recognition-applications-and-code-examples-47f38642ff20

"""
git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git
cd GOT-OCR2.0

conda create -n got python=3.10 -y
conda activate got

pip install -e .

pip install ninja
pip install flash-attn --no-build-isolation
"""

# Part 1: Running the OCR Script and Capturing Raw Output
import os
import subprocess

def run_ocr(client, model_name, image_path):
    # Define the OCR script directory
    ocr_dir = "/path/to/GOT-OCR2.0/GOT-OCR-2.0-master"
    
    # Get the relative path of the image file
    rel_image_path = os.path.relpath(image_path, ocr_dir)

    command = [
        "python3",
        "GOT/demo/run_ocr_2.0.py",
        "--model-name",
        "./GOT_weights/",
        "--image-file",
        rel_image_path,
        "--type",
        "ocr"
    ]
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            cwd=ocr_dir
        )
        raw_output = result.stdout.strip()

# Part 2: Extracting OCR Text and Preparing the Prompt
def extract_ocr_text(raw_output):
    # Find the OCR text starting from "<|im_start|>assistant\n\n"
    match = re.search(r'<\|im_start\|>assistant\n\n(.*?)(?=<\|im_end\|>|$)', raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "No OCR text found"

# Extract the OCR text
ocr_text = extract_ocr_text(raw_output)  # You'll need to implement this function

prompt = f"""
The following text is raw OCR output from an image. Please extract any meaningful text or code snippets from it, 
ignoring any noise or irrelevant information. If it's code, format it properly. If it's text, clean it up and 
present it in a readable format.

Raw OCR output:
{ocr_text}

Output should include ONLY extracted content. No intro, no outro, no formatting, no comments, no explanations, no nothing:
"""

# Part 3: Using the Language Model to Clean Up Text and Returning Results
    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    
    extracted_content = message.content[0].text.strip()
    
    return {
        "raw_ocr": ocr_text,
        "extracted_content": extracted_content
    }
except subprocess.CalledProcessError as e:
    print(f"Error running OCR on {image_path}: {e}")
    return None


