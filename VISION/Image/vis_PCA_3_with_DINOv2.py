## From https://jvgd.medium.com/dinov2-visualization-4a9df1a42387

# Dependencies
# torch==2.3.1
# torchvision==0.18.1
# einops==0.7.0

import torch
from einops import rearrange
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from torchvision.io.image import read_image, ImageReadMode

# Reading images and making sure they have same shape
I1 = read_image("images/image_1.png", ImageReadMode.RGB)
I2 = read_image("images/image_2.png", ImageReadMode.RGB)

H, W = 672, 672
I1 = resize(I1, (H, W))
I2 = resize(I2, (H, W))

I = torch.stack([I1, I2], dim=0)

# Mean & Std for tensors normalized to [0, 1]
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

I_norm = norm(I / 255)

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
features = dinov2.forward_features(I_norm)
E_patch = features["x_norm_patchtokens"]

E_patch_norm = rearrange(E_patch, "B L E -> (B L) E")

# Getting Values of the pricipal value decomposition
_, _, V = torch.pca_lowrank(E_patch_norm)

# Projecting embeddings to the first component of the V matrix
E_pca_1 = torch.matmul(E_patch_norm, V[:, :1])


def minmax_norm(x):
    """Min-max normalization"""
    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
    
E_pca_1_norm = minmax_norm(E_pca_1)

M_fg = E_pca_1_norm.squeeze() > 0.5
M_bg = E_pca_1_norm.squeeze() <= 0.5 

# Getting Values of the pricipal value decomposition for foreground pixels
_, _, V = torch.pca_lowrank(E_patch_norm[M_fg])

# Projecting foreground embeddings to the first 3 component of the V matrix
E_pca_3_fg = torch.matmul(E_patch_norm[M_fg], V[:, :3])
E_pca_3_fg = minmax_norm(E_pca_3_fg)

B, L, _ = E_patch.shape
Z = B * L
I_draw = torch.zeros(Z,3)

I_draw[M_fg] = E_pca_3_fg

I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)

I_draw = rearrange(I_draw, "B (h w) C -> B h w C", h=H//14, w=W//14)

# Unpacking PCA images
image_1_pca = I_draw[0]
image_2_pca = I_draw[1]

# To chanel first format torchvision format
image_1_pca = rearrange(image_1_pca, "H W C -> C H W")
image_2_pca = rearrange(image_2_pca, "H W C -> C H W")

# Resizing it to ease visualization 
image_1_pca = resize(image_1_pca, (H,W))
image_2_pca = resize(image_2_pca, (H,W))

# Saving
save_image(image_1_pca, "images/image_1_pca.png")
save_image(image_2_pca, "images/image_2_pca.png")
