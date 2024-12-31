### From https://medium.com/@alejandro7899871776/point-and-count-objects-using-small-vlms-on-your-local-machine-3a769c7f2b6c

pip install -U mlx-vlm
pip install einops
pip install torch torchvision
pip install matplotlib

from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
import matplotlib.pyplot as plt

model, processor = load("mlx-community/Molmo-7B-D-0924-4bit",processor_config={"trust_remote_code": True})
config = model.config

image_path = "pipes_test.jpg"
image = load_image(image_path)

messages = [{"role": "user", "content": "Point the pipes in the images"}]

prompt = apply_chat_template(processor, config, messages)

output = generate(model, processor, image, prompt, max_tokens=1200, temperature=0.7)

def parse_points(points_str):
    # Function was taken from https://github.com/Blaizzy/mlx-vlm
    if isinstance(points_str, tuple):
        return points_str

    x_coords = []
    y_coords = []

    # Handle multi-point format
    if 'x1="' in points_str:
        i = 1
        while True:
            try:
                x = float(points_str.split(f'x{i}="')[1].split('"')[0])
                y = float(points_str.split(f'y{i}="')[1].split('"')[0])
                x_coords.append(x)
                y_coords.append(y)
                i += 1
            except IndexError:
                break
    elif 'x="' in points_str:
        x = float(points_str.split('x="')[1].split('"')[0])
        y = float(points_str.split('y="')[1].split('"')[0])
        x_coords.append(x)
        y_coords.append(y)

    try:
        labels = points_str.split('alt="')[1].split('">')[0].split(", ")
        item_labels = labels
    except IndexError:
        item_labels = [f"Point {i+1}" for i in range(len(x_coords))]

    return x_coords, y_coords, item_labels

def plot_locations(points: str | tuple, image, point_size=10, font_size=12):
    if isinstance(points, str):
        x_coords, y_coords, item_labels = parse_points(points)
    else:
        x_coords, y_coords, item_labels = points

    grayscale_image = image.convert("L")

    img_width, img_height = grayscale_image.size

    x_norm = [(x / 100) * img_width for x in x_coords]
    y_norm = [(y / 100) * img_height for y in y_coords]

    if len(item_labels) != len(x_norm):
        item_labels *= len(x_norm)

    plt.figure(figsize=(10, 8))
    plt.imshow(grayscale_image, cmap="gray")

    plt.axis("off")

    for i, (x, y, label) in enumerate(zip(x_norm, y_norm, item_labels), start=1):
        label_with_number = f"{i}"
        plt.plot(x, y, "o", color="red", markersize=point_size, label=label_with_number)

        plt.annotate(
            label_with_number,
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="red",
            fontsize=font_size,
        )

    plt.show()

