#!/usr/bin/env python3
"""
Convert hand_poses data to proper sequence format expected by RealDexDataset.
Each dataset should have all its samples in a single file, not individual files.
"""

import numpy as np
import os
import glob
from tqdm import tqdm

def convert_hand_poses_to_qpos_format(hand_poses):
    """
    Convert 28-dim hand_poses to separate components:
    - hand_transl: 3-dim (translation)
    - hand_orient: 3-dim (rotation, axis-angle)
    - qpos: 22-dim (joint angles)
    """
    # hand_poses is [N, 28]
    # Split: [translation(3), rotation(3), qpos(22)]
    hand_transl = hand_poses[:, :3]  # First 3: translation
    hand_orient = hand_poses[:, 3:6]  # Next 3: rotation (axis-angle)
    qpos = hand_poses[:, 6:]  # Last 22: joint angles
    
    return hand_transl, hand_orient, qpos

def convert_dataset_to_sequence(input_file, output_file):
    """Convert a single dataset to sequence format expected by RealDexDataset."""
    print(f"Converting {input_file} to sequence format...")
    
    # Load original data
    data = np.load(input_file, allow_pickle=True)
    
    # Extract components
    hand_poses = data['hand_poses']
    obj_pc = data['obj_pc']
    
    # Convert hand_poses to separate components
    hand_transl, hand_orient, qpos = convert_hand_poses_to_qpos_format(hand_poses)
    
    print(f"  Original hand_poses shape: {hand_poses.shape}")
    print(f"  Converted - qpos: {qpos.shape}, hand_transl: {hand_transl.shape}, hand_orient: {hand_orient.shape}")
    
    # Create object pose data (using identity for now, as we don't have object tracking)
    num_samples = hand_poses.shape[0]
    object_transl = np.zeros((num_samples, 3))  # Object at origin
    object_orient = np.tile(np.eye(3), (num_samples, 1, 1))  # Identity rotation
    
    # For object points, use the first sample's point cloud for all samples
    # The dataset expects object_points to be [num_points, 3] for the object
    object_points = obj_pc[0]  # Use first sample's point cloud
    object_colors = np.ones_like(object_points) * 0.5  # Default gray color
    
    # Create sequence data dict (all samples in one file)
    sequence_data = {
        'qpos': qpos,  # [num_samples, 22]
        'hand_transl': hand_transl,  # [num_samples, 3]
        'hand_orient': hand_orient,  # [num_samples, 3]
        'object_transl': object_transl,  # [num_samples, 3]
        'object_orient': object_orient,  # [num_samples, 3, 3]
        'object_points': object_points,  # [num_points, 3]
        'object_colors': object_colors   # [num_points, 3]
    }
    
    # Save as sequence file
    np.savez(output_file, **sequence_data)
    print(f"  Saved sequence with {num_samples} samples to {output_file}")
    return num_samples

def main():
    """Convert all datasets to sequence format."""
    
    # Input and output directories
    input_base = "./data/your_data_converted"
    output_base = "./data/sequence_format"
    
    # Find all datasets
    dataset_dirs = [d for d in os.listdir(input_base) 
                   if os.path.isdir(os.path.join(input_base, d))]
    
    print(f"Found {len(dataset_dirs)} datasets to convert: {dataset_dirs}")
    
    os.makedirs(output_base, exist_ok=True)
    
    total_samples = 0
    for dataset_name in dataset_dirs:
        input_file = os.path.join(input_base, dataset_name, "converted_data.npz")
        output_file = os.path.join(output_base, f"{dataset_name}.npz")
        
        if os.path.exists(input_file):
            samples = convert_dataset_to_sequence(input_file, output_file)
            total_samples += samples
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    print(f"\nConversion complete! Total samples: {total_samples}")
    print(f"Output directory: {output_base}")

if __name__ == "__main__":
    main()