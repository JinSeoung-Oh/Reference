"""
### https://medium.com/data-science-collective/from-images-to-semantic-3d-gaussian-splatting-with-python-complete-guide-ff9d3d240847
"""

conda create -n da3_env python=3.10
conda activate da3_env

# The Essentials
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy matplotlib
pip install open3d

# The Star of the Show
# We install directly from the source to get the latest V3 API
pip install git+https://github.com/LiheYoung/Depth-Anything

# Optional: get an IDE
pip install spyder

import glob
import os
import torch
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from depth_anything_3.api import DepthAnything3
from sklearn.neighbors import KDTree

def setup_paths(data_folder="SAMPLE_SCENE"):
    """Create project paths for data, results, and models"""
    paths = {
        'data': f"../../DATA/{data_folder}",
        'results': f"../../RESULTS/{data_folder}",
        'masks': f"../../RESULTS/{data_folder}/masks"
    }
    os.makedirs(paths['results'], exist_ok=True)
    os.makedirs(paths['masks'], exist_ok=True)
    return paths

def visualize_depth_and_confidence(images, depths, confidences, sample_idx=0):
    """Show RGB image, depth map, and confidence map side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(images[sample_idx])
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    axes[1].imshow(depths[sample_idx], cmap='turbo')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    axes[2].imshow(confidences[sample_idx], cmap='viridis')
    axes[2].set_title('Confidence Map')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_point_cloud_open3d(points, colors=None, window_name="Point Cloud"):
    """Display 3D point cloud with Open3D viewer"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

# Florent's Note: These visualization functions are reusable across all steps

def load_da3_model(model_name="depth-anything/DA3NESTED-GIANT-LARGE"):
    """Initialize Depth-Anything-3 model on available device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    
    return model, device

# Time to test step 1: Load the DA3 model
model, device = load_da3_model()
print("DA3 model loaded successfully")

def load_images_from_folder(data_path, extensions=['*.jpg', '*.png', '*.jpeg']):
    """Scan folder and load all images with supported extensions"""
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(glob.glob(os.path.join(data_path, ext))))
    
    print(f"Found {len(image_files)} images in {data_path}")
    return image_files

----------------------------------------
paths = setup_paths("CAR")
print(f"Project paths created: {paths}")
image_files = load_images_from_folder(paths['data'])

# %% Step 3: Run DA3 Inference for Depth and Poses
def run_da3_inference(model, image_files, process_res_method="upper_bound_resize"):
    """Run Depth-Anything-3 to get depth maps, camera poses, and intrinsics"""
    prediction = model.inference(
        image=image_files,
        infer_gs=True,
        process_res_method=process_res_method
    )
    
    print(f"Depth maps shape: {prediction.depth.shape}")
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"Confidence shape: {prediction.conf.shape}")
    
    return prediction

prediction = run_da3_inference(model, image_files)
visualize_depth_and_confidence(
    prediction.processed_images, 
    prediction.depth, 
    prediction.conf, 
    sample_idx=0
)


X=(u−cx)⋅Z/fx X = (u - c_x) \cdot Z / f_x X=(u−cx​)⋅Z/fx​
Y=(v−cy)⋅Z/fy Y = (v - c_y) \cdot Z / f_y Y=(v−cy​)⋅Z/fy
​
# %% Step 4: Generate 3D Point Cloud from Depth Maps
def depth_to_point_cloud(depth_map, rgb_image, intrinsics, extrinsics, conf_map=None, conf_thresh=0.5):
    """Back-project depth map to 3D points using camera parameters"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter by confidence if provided
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u, v, depth_map, rgb_image = u[valid_mask], v[valid_mask], depth_map[valid_mask], rgb_image[valid_mask]
    else:
        u, v, depth_map = u.flatten(), v.flatten(), depth_map.flatten()
        rgb_image = rgb_image.reshape(-1, 3)
    
    # Back-project to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates using extrinsics (w2c format)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R  # Inverse transform
    
    colors = rgb_image.astype(np.float32) / 255.0
    return points_world, colors

def merge_point_clouds(prediction, conf_thresh=0.5):
    """Combine all frames into single point cloud"""
    all_points = []
    all_colors = []
    
    n_frames = len(prediction.depth)
    
    for i in range(n_frames):
        points, colors = depth_to_point_cloud(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            conf_thresh
        )
        all_points.append(points)
        all_colors.append(colors)
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"Merged point cloud: {len(merged_points)} points")
    return merged_points, merged_colors

# Time to Generate 3D point cloud
points_3d, colors_3d = merge_point_clouds(prediction, conf_thresh=0.4)
visualize_point_cloud_open3d(points_3d, colors_3d, window_name="Full Scene Point Cloud")
    

# %% Step 4b: Cleaning the 3D Geometry [OPTIONAL]
import time

def clean_point_cloud_open3d(points_3d, colors_3d, nb_neighbors=20, std_ratio=2.0):
    """
    Cleans a point cloud using Statistical Outlier Removal (SOR) via Open3D.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is not installed. Run `pip install open3d` or use the scipy version.")

    # 1. Convert Numpy to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Open3D expects colors in float [0, 1]. If inputs are int [0, 255], normalize them.
    if colors_3d.max() > 1.0:
        pcd.colors = o3d.utility.Vector3dVector(colors_3d / 255.0)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors_3d)

    # 2. Run SOR (Implemented in optimized C++)
    # cl: The cleaned point cloud object
    # ind: The indices of the points that remain
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)

    # 3. Use the indices to filter the original numpy arrays 
    # (This ensures we preserve exact original data types/values if needed)
    inlier_mask = np.asarray(ind)
    
    cleaned_points = points_3d[inlier_mask]
    cleaned_colors = colors_3d[inlier_mask]

    return cleaned_points, cleaned_colors

def clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=2.0):
    """
    Cleans a point cloud using SOR via Scipy cKDTree.
    """
    from scipy.spatial import cKDTree

    # 1. Build KD-Tree
    tree = cKDTree(points_3d)

    # 2. Query neighbors
    # k needs to be nb_neighbors + 1 because the point itself is included in results
    distances, _ = tree.query(points_3d, k=nb_neighbors + 1, workers=-1) 
    
    # Exclude the first column (distance to self, which is 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # 3. Calculate statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    # 4. Generate Mask
    distance_threshold = global_mean + (std_ratio * global_std)
    mask = mean_distances < distance_threshold

    return points_3d[mask], colors_3d[mask]

# --- Demo Usage ---
if __name__ == "__main__":

    # Try Open3D method
    try:
        start = time.time()
        clean_pts, clean_cols = clean_point_cloud_open3d(points_3d, colors_3d)
        end = time.time()
        print(f"\n[Open3D] Cleaned shape: {clean_pts.shape}")
        print(f"[Open3D] Time taken: {end - start:.4f} seconds")
    except ImportError:
        print("\n[Open3D] Skipped (library not found)")

    # Try Scipy method
    start = time.time()
    clean_pts_sci, clean_cols_sci = clean_point_cloud_scipy(points_3d, colors_3d)
    end = time.time()
    print(f"\n[Scipy] Cleaned shape: {clean_pts_sci.shape}")
    print(f"[Scipy] Time taken: {end - start:.4f} seconds")
