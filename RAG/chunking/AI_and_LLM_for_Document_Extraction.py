## From https://medium.com/@krtarunsingh/ai-and-llm-for-document-extraction-simplifying-complex-formats-with-ease-b3261b5be58e

!pip install Pillow torch torchvision transformers sentencepiece pymupdf

import torch
from transformers import AutoModel, AutoTokenizer
import fitz  # PyMuPDF
from PIL import Image

# Load the model and tokenizer
model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device="cuda")

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)
model.eval()

# Open the PDF file
pdf_path = "mypdf.pdf"
pdf_document = fitz.open(pdf_path)

# Store images
images = []

# Loop through each page and convert it to an image
for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(img)

pdf_document.close()

question = """Extract all the text in this image.
If there is a header or a footer, just ignore it.
Extract tables as markdown tables.
Don't use the subtitles for the list items, just return the list as text."""

msgs = [{"role": "user", "content": question}]

res = model.chat(
    image=images[0],  # Using the first image as an example
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)

print(res)
