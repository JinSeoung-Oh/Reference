## From https://pub.towardsai.net/explainable-ai-for-clip-the-architecture-explanation-and-its-application-for-segment-anything-b78ad5f05bb6

"""
1. Explainability in AI and CLIP Surgery Overview
   Explainability is a critical topic in AI models due to the increasingly complex and opaque nature of many systems.
   CLIP, developed by OpenAI, is known for its zero-shot image classification abilities, utilizing image-text pairs to align embeddings through contrastive learning. 
   However, despite its impressive capabilities, understanding why CLIP produces certain results is challenging.

   The paper titled “CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks” explores techniques to make CLIP more interpretable. 
   The blog delves into how CLIP Surgery enhances explainability and applies it to real-world tasks.

   - 1. Recap of CLIP
        CLIP combines image and text encoders, trained on pairs like images of dogs and the corresponding text "The photo of a dog." 
        It uses a contrastive learning approach, maximizing the similarity between correct pairs while minimizing others, 
        resulting in embeddings that enable tasks like zero-shot classification. However, training CLIP requires substantial resources, 
        with OpenAI utilizing 592 GPUs over 18 days.

   - 2. Explanation of CLIP Surgery Algorithm
        CLIP Surgery aims to improve the explainability of CLIP without additional training. 
        It visualizes the activation maps, making it possible to see which regions correspond to specific labels. 
        The key innovation involves modifications to CLIP’s attention layers:

        - The original query-key self-attention visualizes irrelevant areas.
        - The new value-value self-attention focuses solely on relevant semantic regions, offering more precise results.

        This approach reveals redundant features across all labels and suggests filtering out common regions to enhance accuracy. 
        The algorithm applies a series of steps to achieve this:

        -1) Calculate a weight vector that equalizes class influence.
        -2) Remove redundant features to obtain purer feature maps.
        -3) Generate a final similarity matrix to visualize the target areas more effectively.
        The result is an activation map that accurately captures the label-specific semantic region, offering a powerful visualization tool.

2. Applications: Real-World Data and Segment Anything
   The paper also applies CLIP Surgery to real-world datasets like Flickr30k and demonstrates its potential in enhancing models like Segment Anything (SAM),
   which is a foundation model for segmentation tasks.

   For real-world data, comparisons show that CLIP Surgery significantly outperforms the original CLIP in detecting relevant objects.
   However, it encounters issues when irrelevant objects are present. Introducing a threshold during normalization can help mitigate this issue.

   For SAM, CLIP Surgery assists in providing accurate points for segmentation tasks by downsampling and analyzing activation maps. 
   By integrating CLIP Surgery with SAM, points are automatically selected, reducing the manual effort typically required. 
   The results show promising accuracy in detecting target labels, though the model's performance still hinges on CLIP’s original understanding capabilitie
"""

import os
import sys
# please append your CLIP_Surgery directory path
sys.path.append('./clip')

import clip
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from clip.clip_surgery_model import CLIPSurgery
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from segment_anything import sam_model_registry, SamPredictor

# Your SAM model path
sam_model_path = 'sam_vit_h_4b8939.pth'
dataset_path = '<Your Flickr dataset path>'

# Candidate labels you want to find in the image
all_texts = ['building', 'cat', 'cloth', 'floor', 'flower', 'light', 'plant', 'road', 'sky', 'window']

class FlickrDataset(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 image_size: int = 224):
        
        self.dataset_path = dataset_path
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                    ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )

        self.data = self.collect_data()

    def __len__(self):
        return len(self.data)

    def collect_data(self):
        files_list = []

        for root, _, files in os.walk(self.dataset_path):
            if len(files) > 0:
                for f in files:
                    if 'jpg' in f:
                        # append an image file path
                        filepath = os.path.join(root, f)
                        files_list.append(filepath)

        return files_list

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data[idx]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            img = self.transform(img)

        sample = {'image': img, 'cv2_image': cv2_img}

        return sample

def extract_CLIP_similaritymap(image: torch.Tensor, model, image_size, threshold: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        
        # Extract image features
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)

        if type(model) == CLIPSurgery:
            # Apply feature surgery
            similarity = clip.clip_feature_surgery(image_features, text_features)
            
        else:
            # Similarity map from image tokens with min-max norm and resize, B,H,W,N
            similarity = image_features @ text_features.t()

        similarity = torch.where(similarity >= threshold, similarity, 0.0)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], image_size)

        return similarity_map

def apply_heatmap(similarity_map: torch.Tensor, cv2_img: np.array) -> np.array:
    """
    Apply heatmap using similarity map to the original image 
    """
    
    vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis = cv2_img * 0.4 + vis * 0.6
    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    
    return vis
  
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/16", device=device)
model.eval()

model_surgery, preprocess = clip.load("CS-ViT-B/16", device=device)
model_surgery.eval()

flicker_dataset = FlickrDataset(dataset_path=dataset_path)

for idx, batch in enumerate(flicker_dataset):

    image = batch['image']
    cv2_image = batch['cv2_image']
    
    image_size = cv2_image.shape[:2]

    similarity_map_raw = extract_CLIP_similaritymap(image=image, model=model, image_size=image_size, threshold=-10.0)
    similarity_map = extract_CLIP_similaritymap(image=image, model=model_surgery, image_size=image_size, threshold=-10.0)

    # Draw similarity map
    for b in range(similarity_map.shape[0]):
        for n in range(similarity_map.shape[-1]):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            label = all_texts[n]
            
            vis_raw = apply_heatmap(similarity_map=similarity_map_raw, cv2_img=cv2_image)
            vis_surgery = apply_heatmap(similarity_map=similarity_map, cv2_img=cv2_image)

            ax[0].imshow(vis_raw)
            ax[0].set_title(f'CLIP query-key attention: {label}')
            ax[0].set_axis_off()
            
            ax[1].imshow(vis_surgery)
            ax[1].set_title(f'CLIP value-value attention: {label}')
            ax[1].set_axis_off()
            
            plt.show()
            
    if idx >= 0:
        break




