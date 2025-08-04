### From https://medium.com/data-science-collective/smart-3d-change-detection-python-tutorial-for-point-clouds-0dfd9945eb6a

!pip install open3d numpy scipy matplotlib laspy

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import laspy
import os

#%% Load Point Cloud Data - handles LAS/LAZ and PLY formats
def load_point_cloud(file_path):
    """
    Load point cloud data from LAS/LAZ or PLY files
    """
    if file_path.endswith('.las') or file_path.endswith('.laz'):
        # Load LAS/LAZ file
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif file_path.endswith('.ply'):
        # Load PLY file directly with Open3D
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded {len(pcd.points)} points from {file_path}")
    return pcd

#%% Step 2. Preprocess Point Clouds - downsampling and outlier removal
def preprocess_point_cloud(pcd, voxel_size=0.05, remove_outliers=True):
    """
    Preprocess point cloud: downsampling and outlier removal
    """
    # Downsample using voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals for the downsampled point cloud
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Remove outliers if requested
    if remove_outliers:
        # Statistical outlier removal
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"Preprocessed cloud has {len(pcd_down.points)} points")
    return pcd_down

#%% Step 3: Register Point Clouds if necessary
def register_point_clouds(source, target, voxel_size=0.05, max_iter=100):
    """
    Register source point cloud to target using point-to-plane ICP
    """
    # Initialize transformation with identity matrix
    init_transformation = np.identity(4)
    
    # Set convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iter
    )
    
    # Point-to-plane ICP registration
    result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size*2, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    # Apply transformation to source
    source_transformed = source.transform(result.transformation)
    
    print(f"Registration finished with fitness: {result.fitness}, RMSE: {result.inlier_rmse}")

#%% Step 4. Compute Cloud-to-Cloud Distance using KD-Tree spatial indexing
def compute_cloud_distances(source, target):
    """
    Compute point-to-point distances between source and target clouds
    """
    # Convert target points to numpy array for KDTree
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    
    # Build KDTree from target points
    tree = KDTree(target_points)
    
    # Query the tree for nearest neighbor distances
    distances, _ = tree.query(source_points)
    
    print(f"Computed distances for {len(source_points)} points")
    return distances

#%% Step 5. Statistical Analysis of Changes with threshold-based detection
def analyze_changes(distances, threshold=0.1):
    """
    Analyze distances to identify significant changes
    """
    # Identify points with distance greater than threshold
    change_indices = np.where(distances > threshold)[0]
    change_distances = distances[change_indices]
    
    # Calculate statistics
    if len(change_distances) > 0:
        mean_change = np.mean(change_distances)
        max_change = np.max(change_distances)
        total_volume_change = len(change_indices) / len(distances)  # Approximate as percentage of points
        
        print(f"Detected {len(change_indices)} points with significant change")
        print(f"Mean change: {mean_change:.3f}m, Max change: {max_change:.3f}m")
        print(f"Approximate volume change: {total_volume_change*100:.2f}%")
        
        return change_indices, {
            "mean_change": mean_change,
            "max_change": max_change,
            "volume_change_percentage": total_volume_change*100
        }
    else:
        print("No significant changes detected")
        return [], {"mean_change": 0, "max_change": 0, "volume_change_percentage": 0}

#%% Create Distance Heatmap with gradient color mapping
def create_distance_heatmap(source, distances):
    """
    Visualize the entire point cloud as a heatmap based on distance values
    """
    # Create a copy of the source cloud
    heatmap_pcd = o3d.geometry.PointCloud()
    heatmap_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.points))
    
    # Normalize distances for visualization
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Create a colormap (blue=close, red=far)
    if max_dist > min_dist:
        normalized_dists = (distances - min_dist) / (max_dist - min_dist)
    else:
        normalized_dists = np.ones_like(distances) * 0.5
    
    # Create color array using a gradient from blue to red
    colors = np.zeros((len(distances), 3))
    colors[:, 0] = normalized_dists  # Red channel increases with distance
    colors[:, 2] = 1 - normalized_dists  # Blue channel decreases with distance
    
    # Add green component for a more dynamic color range
    colors[:, 1] = np.where(normalized_dists < 0.5, 
                           normalized_dists * 2, 
                           (1 - normalized_dists) * 2)
    
    heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Heatmap color scale: Blue = {min_dist:.3f}m, Red = {max_dist:.3f}m")
    
    return heatmap_pcd

#%% Step 6. Object-based Change-detection using region growing algorithms
def detect_missing_regions(source, target, distances, distance_threshold=0.1, region_size_threshold=10):
    """
    Detect regions in source that have no correspondence in target using distance thresholding and region growing
    """
    source_points = np.asarray(source.points)
    
    # Find points that are far from any point in the target (potentially missing)
    missing_indices = np.where(distances > distance_threshold)[0]
    
    if len(missing_indices) == 0:
        print("No significant differences detected")
        return [], [], []
    
    # Create a KDTree of the source points
    source_tree = KDTree(source_points)
    
    # Initialize variables for region growing
    all_regions = []
    processed = np.zeros(len(source_points), dtype=bool)
    
    # Process each unprocessed missing point
    for idx in missing_indices:
        if processed[idx]:
            continue
            
        # Start a new region with this point
        current_region = [idx]
        processed[idx] = True
        
        # Grow the region
        i = 0
        while i < len(current_region):
            # Get current point in the growing region
            current_idx = current_region[i]
            
            # Find neighbors within the 3D neighborhood (using KDTree)
            neighbors_dist, neighbors_idx = source_tree.query(
                source_points[current_idx].reshape(1, -1), 
                k=20  # Consider 20 nearest neighbors
            )
            
            # Add unprocessed neighbors that are also missing
            for neighbor_idx in neighbors_idx[0][1:]:  # Skip the point itself
                if not processed[neighbor_idx] and neighbor_idx in missing_indices:
                    current_region.append(neighbor_idx)
                    processed[neighbor_idx] = True
            
            i += 1
        
        # Store region if it's large enough (to filter out noise)
        if len(current_region) >= region_size_threshold:
            all_regions.append(current_region)
    
    # Flatten all regions into a single list of indices
    all_missing_indices = []
    region_labels = np.zeros(len(source_points), dtype=int)
    
    for region_idx, region in enumerate(all_regions, 1):
        all_missing_indices.extend(region)
        # Label each point with its region number
        for point_idx in region:
            region_labels[point_idx] = region_idx
    
    print(f"Detected {len(all_regions)} missing regions with total {len(all_missing_indices)} points")
    
    # Return missing regions, all missing indices, and region labels
    return all_regions, np.array(all_missing_indices), region_labels

#%% Step 7. Upsampling predictions to full point cloud resolution
def transfer_colors_to_original(original_pcd, colored_downsampled_pcd):

    # Create a copy of the original point cloud to add colors to
    colored_original = o3d.geometry.PointCloud()
    colored_original.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points))
    
    # Get numpy arrays of points
    original_points = np.asarray(original_pcd.points)
    downsampled_points = np.asarray(colored_downsampled_pcd.points)
    downsampled_colors = np.asarray(colored_downsampled_pcd.colors)
    
    # Build KDTree from downsampled points for nearest neighbor search
    tree = KDTree(downsampled_points)
    
    # For each point in the original cloud, find the nearest neighbor in the downsampled cloud
    _, indices = tree.query(original_points)
    
    # Transfer colors based on nearest neighbor relationship
    original_colors = downsampled_colors[indices]
    colored_original.colors = o3d.utility.Vector3dVector(original_colors)
    
    print(f"Transferred colors from downsampled cloud ({len(downsampled_points)} points) to original cloud ({len(original_points)} points)")
    
    return colored_original

def save_results(colored_source, missing_pcd, heatmap_pcd, regions, stats, output_dir="./output"):
    """
    Save all visualization and analysis results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the colored point cloud
    if colored_source is not None:
        o3d.io.write_point_cloud(f"{output_dir}/colored_source.ply", original_colored)
        print(f"Saved colored source point cloud to {output_dir}/colored_source.ply")
    
    # Save the missing regions point cloud
    if missing_pcd is not None and len(missing_pcd.points) > 0:
        o3d.io.write_point_cloud(f"{output_dir}/missing_regions.ply", missing_pcd)
        print(f"Saved missing regions point cloud to {output_dir}/missing_regions.ply")
    
    # Save the heatmap point cloud
    if heatmap_pcd is not None:
        o3d.io.write_point_cloud(f"{output_dir}/distance_heatmap.ply", heatmap_pcd)
        print(f"Saved distance heatmap to {output_dir}/distance_heatmap.ply")
    
    # Save region statistics
    if regions:
        with open(f"{output_dir}/region_stats.txt", "w") as f:
            f.write(f"Total number of detected regions: {len(regions)}\n\n")
            for i, region in enumerate(regions):
                f.write(f"Region {i+1}: {len(region)} points\n")
        print(f"Saved region statistics to {output_dir}/region_stats.txt")
    
    # Save overall statistics
    with open(f"{output_dir}/change_stats.txt", "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved overall statistics to {output_dir}/change_stats.txt")
