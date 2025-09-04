#!/usr/bin/env python3
"""
Convert official RealDex final_data.npy files to proper qpos format for training.
The official RealDex data already contains qpos, but needs to be reformatted.
"""

import numpy as np
import os
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def convert_realdex_dataset(input_file, output_dir):
    """Convert official RealDex final_data.npy to qpos format."""
    print(f"Converting {input_file}...")
    
    # Load official RealDex data
    data = np.load(input_file, allow_pickle=True).item()
    
    # Extract components from official format
    qpos = data['qpos']  # [N, 22] - already in correct format
    hand_transl = data['global_transl']  # [N, 3] - hand translation
    hand_orient_mat = data['global_orient']  # [N, 3, 3] - rotation matrices
    object_transl = data['object_transl']  # [N, 3] - object translation
    object_orient_mat = data['object_orient']  # [N, 3, 3] - object rotation matrices
    
    # Convert rotation matrices to axis-angle format for hand_orient
    num_samples = qpos.shape[0]
    hand_orient = np.zeros((num_samples, 3))
    for i in range(num_samples):
        rot = R.from_matrix(hand_orient_mat[i])
        hand_orient[i] = rot.as_rotvec()  # axis-angle representation
    
    print(f"  Qpos shape: {qpos.shape}")
    print(f"  Hand_transl shape: {hand_transl.shape}")
    print(f"  Hand_orient shape: {hand_orient.shape}")
    print(f"  Object_transl shape: {object_transl.shape}")
    print(f"  Object_orient shape: {object_orient_mat.shape}")
    
    # Create object points and colors (placeholder since we don't have point cloud in final_data)
    # Use a simple cube as placeholder object geometry
    object_points = np.array([
        [-0.05, -0.05, -0.05], [0.05, -0.05, -0.05], [0.05, 0.05, -0.05], [-0.05, 0.05, -0.05],
        [-0.05, -0.05, 0.05], [0.05, -0.05, 0.05], [0.05, 0.05, 0.05], [-0.05, 0.05, 0.05]
    ])
    object_colors = np.ones_like(object_points) * 0.7  # Light gray color
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each sample as a separate file (matching the expected qpos format structure)
    for i in range(num_samples):
        output_file = os.path.join(output_dir, f"{i}.npz")
        
        # Create data dict for this sample
        sample_data = {
            'qpos': qpos[i:i+1],  # Keep batch dimension [1, 22]
            'hand_transl': hand_transl[i:i+1],  # [1, 3]
            'hand_orient': hand_orient[i:i+1],  # [1, 3] - axis-angle
            'object_transl': object_transl[i:i+1],  # [1, 3]
            'object_orient': object_orient_mat[i:i+1],  # [1, 3, 3] - keep as rotation matrix
            'object_points': object_points,  # [num_points, 3]
            'object_colors': object_colors   # [num_points, 3]
        }
        
        np.savez(output_file, **sample_data)
        
    print(f"  Saved {num_samples} samples to {output_dir}")
    return num_samples

def main():
    """Convert all official RealDex datasets to qpos format."""
    
    # Input and output directories
    input_base = "data/realdex_data/storage/group/4dvlab/youzhuo/bags"
    output_base = "data/realdex_qpos_format"
    
    # Find all final_data.npy files
    final_data_files = []
    for root, dirs, files in os.walk(input_base):
        for file in files:
            if file == "final_data.npy":
                final_data_files.append(os.path.join(root, file))
    
    print(f"Found {len(final_data_files)} RealDex datasets with final_data.npy:")
    
    total_samples = 0
    converted_datasets = []
    
    for input_file in final_data_files:
        # Extract dataset name from path
        # e.g., .../air_duster/air_duster_1_20240106/final_data.npy -> air_duster_1_20240106
        path_parts = input_file.split(os.sep)
        object_name = path_parts[-3]  # air_duster
        experiment_name = path_parts[-2]  # air_duster_1_20240106
        dataset_name = f"{object_name}_{experiment_name}"
        
        print(f"\nProcessing: {dataset_name}")
        output_dir = os.path.join(output_base, dataset_name)
        
        try:
            samples = convert_realdex_dataset(input_file, output_dir)
            total_samples += samples
            converted_datasets.append(dataset_name)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    print(f"\nConversion complete!")
    print(f"Total datasets converted: {len(converted_datasets)}")
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {output_base}")
    print(f"Converted datasets: {converted_datasets}")

if __name__ == "__main__":
    main()