## See more detail : From https://generativeai.pub/data-augmentation-for-text-image-multimodel-transformers-c959671e1935

!pip install albumentations==1.4.14 datasets==2.21.0 pdf2image==1.17.0 nltk

### Extract images from pdfs and modify metadata
from datasets import load_dataset 
import json
from pdf2image import convert_from_bytes

dataset = load_dataset('pixparse/pdfa-eng-wds', streaming=True)

for count, sample1 in enumerate(dataset['train']):
    
    pdf = convert_from_bytes(sample1['pdf'], dpi=300)

    bbox_update = lambda x, y, w, h : (x, y, x+w, y+h)

    for i, page in enumerate(pdf):
        page.save(f"data/{count}_{i}.png")
        meta_lines = sample1['json']['pages'][i]['lines']
        result = {k:meta_lines[k] for k in ["text", "bbox"]}

        result = [
            {
                "text" : result['text'][i], 
                "bbox" : bbox_update(*result['bbox'][i])
            } 
            for i in range(len(result['text']))
        ]
        with open(f"data/{count}_{i}.json", "w") as _f:
            json.dump(result, _f, indent=4)

    if count >= 5: break

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import json
import nltk

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stops = stopwords.words('english')

def load_img_and_meta(file_path):
    img = cv2.imread(file_path + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(file_path + ".json") as _f:
        metadata = json.load(_f)

    return img, metadata

def perform_and_plot(transformation):
    f, axs = plt.subplots(1, 3, figsize=(20, 15))
    f.patch.set_linewidth(1)
    f.patch.set_edgecolor('black')
    for i in range(3):
        augmented = transformation(image=img, textimage_metadata=metadata)
        ax = axs[i]
        ax.imshow(augmented['image'])
        ax.axis("off")

img, metadata = load_img_and_meta("data/0_1")

#### Starting of an augmentation methods
# *1. Random Deletion

font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

delete_transform = A.Compose([
    A.TextImage(
        augmentations=['deletion'],
        stopwords=stops,
        font_path=font_path, 
        font_color = ['red', 'blue', 'green', 'indigo'],
        clear_bg=True,
        fraction_range=(0.5, 0.8),
        p=1
    )
])
perform_and_plot(delete_transform)

# *2. Random Swap
font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

swap_transform = A.Compose([
    A.TextImage(
        augmentations=['swap'],
        stopwords=stops,
        font_path=font_path, 
        font_color = ['red', 'blue', 'green', 'indigo'],
        clear_bg=True,
        fraction_range=(0.5, 0.8),
        p=1
    )
])
perform_and_plot(swap_transform)

# *3. Random Insertion
font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

insertion_transform = A.Compose([
    A.TextImage(
        augmentations=['insertion'],
        stopwords=stops,
        font_path=font_path, 
        font_color = ['red', 'blue', 'green', 'indigo'],
        clear_bg=True,
        fraction_range=(0.1, 0.3),
        p=1
    )
])
perform_and_plot(insertion_transform)

# *4. Mix Transformation
font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

mix_transform = A.Compose([
    A.TextImage(
        augmentations=['insertion', 'swap', 'deletion'],
        stopwords=stops,
        font_path=font_path, 
        font_color = ['red', 'blue', 'green', 'indigo'],
        clear_bg=True,
        fraction_range=(0.5, 0.8),
        p=1
    ),
    A.ChannelShuffle(p=0.5),
    A.Affine(p=0.3)
])
perform_and_plot(mix_transform)








