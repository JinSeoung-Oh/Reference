### From https://medium.com/data-science-collective/smoldocling-a-new-era-in-document-processing-3e9b044eeb4a

'''
1. Core Features
   -a. Compact Yet Powerful
       -1. Derived from Hugging Face’s SmolVLM-256M, SmolDocling packs a competitive performance into a very small 
           model size.
       -2. Despite its compactness, it reliably competes with much larger vision models.
2. Document Structure Representation with DocTags
   -a. DocTags Format: An XML-like markup language designed specifically for document transformation.
       -1. Element Type: Defines various content components such as text, image, table, code, title, footnote, etc.
       -2. Position on Page: Uses position tags to indicate the bounding box of each element on the page 
                             in the format <loc_x1><loc_y1><loc_x2><loc_y2>, where:
           -1) x1, y1: Coordinates of the upper left corner.
           -2) x2, y2: Coordinates of the lower right corner.
       -3. Content: Contains the actual text or structural data.
   -b. Nested Structures:
       1. Allows embedding of additional information such as captions for images/tables, hierarchical list items, 
          and classification details for code blocks.
   -c. Advantages:
       -1. Reduced Ambiguity: Clear tag structure minimizes uncertainties.
       -2. Separation of Structure and Content: Distinguishes layout from textual information.
       -3. Preservation of Page Layout: Captures spatial relationships.
       -4. Token Optimization: More efficient processing.
       -5. Enhanced Modeling Performance: Provides consistent, well-structured data to improve learning and 
                                          output generation.
3. End-to-End Architectural Structure
   -a. Input Phase: Accepts page images along with text prompts (e.g., “Convert to Docling”) that guide 
                    the conversion.
   -b. Visual Processing Phase:
       -1. A vision encoder converts page images into visual embeddings.
       -2. Projection and pooling operations compact these embeddings.
   -c. Embedding Integration Phase:
       -1. Merges the projected visual embeddings with textual embeddings derived from prompts.
   -d. Output Generation Phase:
       -1. An LLM (Language Model) processes the combined embeddings to produce output in DocTags format.
   This integrated pipeline ensures that both the visual and textual aspects of documents are accurately captured and represented.

4. Application Areas
   SmolDocling is versatile, supporting tasks such as:
   -a. Document Classification: Automatically categorizes documents.
   -b. Optical Character Recognition (OCR): Converts printed or handwritten text into machine-readable text.
   -c. Layout Analysis: Understands and distinguishes different sections of a document.
   -d. Table Recognition: Extracts and maintains the structure of tabular data.
   -e. Key-Value Extraction: Identifies and extracts paired information.
   -f. Graph Understanding and Equation Recognition: Processes complex elements like code lists, graphs, 
                                                     and mathematical expressions.

5. Performance and Evaluation
   -a. Evaluated on the DocLayNet dataset, SmolDocling shows excellent metrics 
       (lowest Edit Distance and highest F1-Score) in full-page conversion, code lists, and equation recognition.
   -b. It even outperforms larger models (e.g., Qwen2.5 VL with 7B parameters), 
       demonstrating the effectiveness of its design and the DocTags format.

6. Integration and Usage
   Below is an example code snippet (do not skip code) that demonstrates how to integrate SmolDocling into 
   a document processing pipeline. 
   The code shows how to load the model, process an image to generate DocTags, and export the resulting document 
   as Markdown and HTML.

''''
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "docling-core",
#     "mlx-vlm",
#     "pillow",
# ]
# ///
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
import requests
from PIL import Image
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate

## Settings
SHOW_IN_BROWSER = True  # Export output as HTML and open in webbrowser.

## Load the model
model_path = "ds4sd/SmolDocling-256M-preview-mlx-bf16"
model, processor = load(model_path)
config = load_config(model_path)

## Prepare input
prompt = "Convert this page to docling."
image = "sample.png"

# Load image resource
if urlparse(image).scheme != "":  # it is a URL
    response = requests.get(image, stream=True, timeout=10)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content))
else:
    pil_image = Image.open(image)

# Apply chat template
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

## Generate output
print("DocTags: \n\n")
output = ""
for token in stream_generate(
    model, processor, formatted_prompt, [image], max_tokens=4096, verbose=False
):
    output += token.text
    print(token.text, end="")
    if "</doctag>" in token.text:
        break
print("\n\n")

# Populate document
doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([output], [pil_image])
# create a docling document
doc = DoclingDocument(name="SampleDocument")
doc.load_from_doctags(doctags_doc)

## Export as any format
# Markdown
print("Markdown: \n\n")
print(doc.export_to_markdown())

# HTML
if SHOW_IN_BROWSER:
    import webbrowser
    out_path = Path("./output.html")
    doc.save_as_html(out_path, image_mode=ImageRefMode.EMBEDDED)
    webbrowser.open(f"file:///{str(out_path.resolve())}")

''''
Conclusion

SmolDocling represents an innovative approach in document understanding, 
combining a compact architecture with robust document structure preservation via the DocTags format. 
Its end-to-end design integrates image understanding and text generation, 
making it highly effective for diverse tasks such as OCR, layout analysis, table recognition, and more. 
With competitive performance on benchmarks and the advantage of being resource-efficient, 
SmolDocling is well-suited for modern, scalable document processing systems—even in environments with 
limited computational resources.
'''''
