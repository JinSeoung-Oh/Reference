### From https://medium.com/@alejandro7899871776/image-segmentation-with-llms-0786454e5c33

import os
import io
import json
import base64

from io import BytesIO
from PIL import Image, ImageColor, ImageFont, ImageDraw
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from pydantic import BaseModel, ValidationError, field_validator, ConfigDict, Field
import numpy as np

load_dotenv()

GEMINI_TIMEOUT_MS = 60 * 1000
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),http_options=HttpOptions(timeout=GEMINI_TIMEOUT_MS))

class SegmentationOutput(BaseModel):
  label: str
  box_2d: list[int,int,int,int] = Field(..., description="yo,xo,y1,x1")
  mask: str

  @field_validator("mask", mode="before")
  def ensure_prefix(cls, png_str: str) -> str:
      prefix = "data:image/png;base64,"
      if not png_str.startswith(prefix):
          raise ValueError(f"bs64_mask must start with '{prefix}'")
      return png_str

class SegmentationItem(SegmentationOutput):
    np_mask: np.array
    model_config = ConfigDict(arbitrary_types_allowed=True)


def validate_json(json_output: str) -> list[SegmentationOutput] | None:
    segmentation_list: list[SegmentationOutput] = []
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            content = "\n".join(lines[i + 1 :])
            content = content.split("```")[0]
            json_output = content
            break

    try:
        json_list = json.loads(json_output)
    except ValueError as e:
        raise ValueError(f"JSON output was wrongly formatted: {e}")

    if not isinstance(json_list, list):
        return None

    for element in json_list:
        try:
            segmentation = SegmentationOutput.model_validate(element)
            segmentation_list.append(segmentation)
        except ValidationError as e:
            print(f"Validation error {e}")

    return segmentation_list or None

def parse_segmentation_masks(
    predicted_str: str, *, img_height: int, img_width: int
) -> list[SegmentationOutput]:
    validated = validate_json(predicted_str)
    print(validated)
    if not validated:
        return []

    results: list[SegmentationOutput] = []
    for item in validated:
        abs_y0 = int(item.box_2d[0] / 1000 * img_height)
        abs_x0 = int(item.box_2d[1]  / 1000 * img_width)
        abs_y1 = int(item.box_2d[2]  / 1000 * img_height)
        abs_x1 = int(item.box_2d[3]  / 1000 * img_width)

        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            print("Invalid bounding box", (item.box_2d))
            continue

        prefix = "data:image/png;base64,"
        png_str = item.mask
        raw_data = base64.b64decode(png_str.removeprefix(prefix))
        pil_mask = Image.open(io.BytesIO(raw_data))

        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height < 1 or bbox_width < 1:
            print("Invalid bounding box")
            continue

        pil_mask = pil_mask.resize(
            (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
        )
        np_mask_full = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask_full[abs_y0:abs_y1, abs_x0:abs_x1] = np.array(pil_mask)

        try:
            seg_item = SegmentationItem(
                label=item.label,
                box_2d=[abs_y0, abs_x0, abs_y1, abs_x1],
                mask=item.mask,
                np_mask=np_mask_full,
            )
            results.append(seg_item)
        except ValidationError as e:
            print("Validation error in final item:", e)
            continue
    return results

def overlay_mask_on_img(
    img: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.7
) -> Image.Image:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    try:
        color_rgb = ImageColor.getrgb(color)
    except ValueError as e:
        raise ValueError(f"Invalid color name '{color}'. Error: {e}")

    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    colored_layer = np.zeros((height, width, 4), dtype=np.uint8)
    mask_logical = mask > 127
    colored_layer[mask_logical] = overlay_color_rgba

    colored_mask = Image.fromarray(colored_layer, "RGBA")
    return Image.alpha_composite(img_rgba, colored_mask)


additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
def plot_segmentation_masks(
    img: Image.Image, segmentation_items: list[SegmentationItem]
) -> Image.Image:
    colors = [
        "red", "green", "blue", "yellow", "orange", "pink", "purple",
        "brown", "gray", "beige", "turquoise", "cyan", "magenta", "lime",
        "navy", "maroon", "teal", "olive", "coral", "lavender", "violet",
        "gold", "silver",
    ] + additional_colors

    font = ImageFont.load_default()

    # Overlay mask using the NumPy array, not the base64 string
    for i, item in enumerate(segmentation_items):
        color = colors[i % len(colors)]
        img = overlay_mask_on_img(img, item.np_mask, color)

    draw = ImageDraw.Draw(img)

    # Draw bounding boxes and labels using box_2d = [y0, x0, y1, x1]
    for i, item in enumerate(segmentation_items):
        color = colors[i % len(colors)]
        y0, x0, y1, x1 = item.box_2d
        draw.rectangle(
            ((x0, y0), (x1, y1)), outline=color, width=4
        )
        if item.label:
            # Position label slightly above top-left corner
            draw.text((x0 + 8, y0 - 20), item.label, fill=color, font=font)

    return img


if __name__ == "__main__":
    image = "image.png"
    query = "Detect all foreign objects in the conveyor belt"
    prompt = f"{query}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."

    im = Image.open(BytesIO(open(image, "rb").read()))
    im.thumbnail([1024,1024], Image.Resampling.LANCZOS)

    # Run model to find segmentation masks
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06",  # "gemini-2.5-flash-preview-05-20"
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            temperature=0.5,
        )
    )

    # Plot
    segmentation_masks = parse_segmentation_masks(response.text, img_height=im.size[1], img_width=im.size[0])
    im = plot_segmentation_masks(im, segmentation_masks)
    im.show()
