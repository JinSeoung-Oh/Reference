### From https://medium.com/@yash9439/decoding-the-scanned-page-best-methods-for-extracting-text-from-complex-scanned-pdfs-143337f2cfbe

## Method 1: Using Vision Language Models (VLMs)
import PIL.Image
import os
import google.generativeai as genai
from pdf2image import convert_from_path

# Replace with your API key
GOOGLE_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)
pdf_path = "test.pdf" # Change this path to point to your pdf
pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
# Create the output directory if it doesn't exist
output_dir = "GeminiResult"
os.makedirs(output_dir, exist_ok=True)
# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")
prompt = """
    Extract all text content and tabular data from this image, strictly preserving the original reading order as they appear on the page.
    1. **Reading Order:** Process the content strictly based on the reading order within the image. Do not rearrange or reorder blocks or tables.
    2. **Text Blocks:** Extract distinct blocks of text and represent each block as a separate entity, separated by double newlines ("\\n\\n").
    3. **Tables:** Identify any tables present in the image. For each table, output it in a structured, comma-separated format (.csv). Each row of the table should be on a new line, with commas separating column values.
        - Include the header row, if present.
        - Ensure that all columns of each row are comma separated values.
    4. **Output Format:**
        - Output text blocks and tables in the order they are read on the page. When a table is encountered while reading the page, output it in CSV format at that point in the output.
    5. If there are no text or no tables return empty string.
     If the table contains only one row, then return text of that row separated by comma.
    """
try:
    # Convert all pages of the PDF to PIL image objects
    images = convert_from_path(pdf_path)
    
    if not images:
        raise FileNotFoundError(f"Could not convert the PDF to images")
    for i, img in enumerate(images):
        page_number = i + 1
        output_file_path = os.path.join(output_dir, f"{pdf_name}_{page_number}.txt")
        
        try:
           response = model.generate_content([prompt, img], generation_config={"max_output_tokens": 4096})
           response.resolve()
           with open(output_file_path, "w", encoding="utf-8") as f:
              f.write(response.text)
           print(f"Processed page {page_number} and saved to {output_file_path}")
        
        except Exception as page_err:
           print(f"Error processing page {page_number}: {page_err}")
           with open(output_file_path, "w", encoding="utf-8") as f:
              f.write(f"Error: An error occurred during processing of page {page_number} : {page_err}")
except FileNotFoundError as e:
    print(f"Error: Could not find file: {e}")
except Exception as e:
    print(f"Error: An error occurred during processing: {e}")


## Method 2: Document Segmentation + OCR for Scalability
from ultralytics import YOLO
import fitz
import os
import pathlib
from PIL import Image, ImageEnhance
import numpy as np
import fitz
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# List of sample PDF files to process
pdf_list = ['test.pdf']
# Load the document segmentation model
docseg_model = YOLO('yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt')
# Initialize a dictionary to store results
mydict = {}

def enhance_image(img):
    """Apply image enhancements for better quality."""
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Enhance color
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    return img

def process_pdf_page(pdf_path, page_num, docseg_model, output_dir):
    """Processes a single page of a PDF with maximum quality settings."""
    
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num]
    
     # Increase the resolution matrix for maximum quality
    zoom = 4  # Increased zoom factor for higher resolution
    matrix = fitz.Matrix(zoom, zoom)
    
    # Use high-quality rendering options
    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,  # Disable alpha channel for clearer images
        colorspace=fitz.csRGB  # Force RGB colorspace
    )
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Apply image enhancements
    img = enhance_image(img)
    
    # Resize with high-quality settings
    if zoom != 1:
        original_size = (int(page.rect.width), int(page.rect.height))
        img = img.resize(original_size, Image.Resampling.LANCZOS)
    # Generate a temporary filename for the page image
    temp_img_filename = os.path.join(output_dir, f"temp_page_{page_num}.png")
    
    # Save with maximum quality settings
    img.save(
        temp_img_filename,
        "PNG",
        quality=100,
        optimize=False,
        dpi=(300, 300)  # Set high DPI
    )
    # Run the model on the image
    results = docseg_model(source=temp_img_filename, save=True, show_labels=True, show_conf=True, boxes=True)
    # Extract the results
    page_width = page.rect.width
    one_third_width = page_width / 3
    
    
    all_coords = []
    
    for entry in results:
        thepath = pathlib.Path(entry.path)
        thecoords = entry.boxes.xyxy.numpy()
        all_coords.extend(thecoords)

    # Sort the coordinates into two groups and then sort each group by y1
    left_group = []
    right_group = []
    for bbox in all_coords:
            x1 = bbox[0]
            if x1 < one_third_width:
                left_group.append(bbox)
            else:
                right_group.append(bbox)

    left_group = sorted(left_group, key=lambda bbox: bbox[1])
    right_group = sorted(right_group, key=lambda bbox: bbox[1])
    
    sorted_coords = left_group + right_group

    mydict[f"{pdf_path} Page {page_num}"] = sorted_coords
    # Clean up the temporary image
    os.remove(temp_img_filename)
    pdf_doc.close()
    
   
# Process each PDF in the list
for pdf_path in pdf_list:
    try:
        pdf_doc = fitz.open(pdf_path)
        num_pages = pdf_doc.page_count
        pdf_doc.close()
        output_dir = os.path.splitext(pdf_path)[0] + "_output"
        os.makedirs(output_dir, exist_ok=True)
        for page_num in range(num_pages):
            process_pdf_page(pdf_path, page_num, docseg_model, output_dir)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
# Create the 'tmp' directory if it doesn't exist
tmp_dir = 'tmp'
os.makedirs(tmp_dir, exist_ok=True)
# Iterate through the results and save cropped images with maximum quality
for key, coords in mydict.items():
    pdf_name, page_info = key.split(" Page ")
    page_number = int(page_info)
    pdf_doc = fitz.open(pdf_name)
    page = pdf_doc[page_number]
    
    zoom = 4
    matrix = fitz.Matrix(zoom,zoom)
    for i, bbox in enumerate(coords):
        # Scale the bounding box coordinates appropriately
        xmin, ymin, xmax, ymax = map(lambda x: x , bbox)
            
        # Create a rectangle from the bounding box
        rect = fitz.Rect(xmin, ymin, xmax, ymax)
            
        # Crop using get_pixmap with a maximum resolution matrix
        cropped_pix = page.get_pixmap(
            clip=rect,
            matrix=matrix,
            alpha=False,
            colorspace=fitz.csRGB
        )
        
        cropped_img = Image.frombytes("RGB", [cropped_pix.width, cropped_pix.height], cropped_pix.samples)
        cropped_img = enhance_image(cropped_img)
        
        output_filename = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(pdf_name))[0]}_page{page_number}_{i}.png")
        
        # Save the cropped image
        cropped_img.save(output_filename, "PNG", quality=100, optimize=False, dpi=(300, 300))
    pdf_doc.close()

def extract_text_from_image(image_path, model):
    """Extracts text from a single image using DocTr."""
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    text_content = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text_content += word.value + " "
            text_content += "\n"
    return text_content.strip()

def process_cropped_images(tmp_dir, pdf_list):
    """Iterates through cropped images, extracts text using DocTr and stores the text in text files."""
    
    doctr_model = ocr_predictor(pretrained=True)
    
    for pdf_path in pdf_list:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_txt_path = f"{pdf_name}_extracted_text.txt"
        
        with open(output_txt_path, 'w', encoding='utf-8') as outfile:
            
            pdf_doc = fitz.open(pdf_path)
            num_pages = pdf_doc.page_count
            pdf_doc.close()
            for page_num in range(num_pages):
                
                outfile.write(f"Page: {page_num}\n")
                
                # Sort filenames of cropped images by chunk order
                cropped_images_for_page = sorted([
                    f for f in os.listdir(tmp_dir)
                    if f.startswith(f"{pdf_name}_page{page_num}_") and f.endswith(".png")
                ], key=lambda f: int(f.split("_")[-1].split(".")[0]))
                
                for i, image_filename in enumerate(cropped_images_for_page):
                    image_path = os.path.join(tmp_dir, image_filename)
                    text = extract_text_from_image(image_path, doctr_model)
                    outfile.write(f"  Chunk {i}: {text}\n")
        print(f"Text extracted from {pdf_name} saved to {output_txt_path}")

# Example usage:
tmp_dir = 'tmp' # Make sure your tmp directory exists
pdf_list = ['test.pdf'] # Your list of PDFs
process_cropped_images(tmp_dir, pdf_list)
