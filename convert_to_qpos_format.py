#!/usr/bin/env python3
"""
Convert hand_poses (28-dim) to proper qpos format for all RealDex datasets.
Based on the analysis of how qpos values are generated in the original pipeline.
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
    
    Based on the format expected by RealDexDataset and the training pipeline.
    """
    # hand_poses is [N, 28]
    # Split: [translation(3), rotation(3), qpos(22)]
    hand_transl = hand_poses[:, :3]  # First 3: translation
    hand_orient = hand_poses[:, 3:6]  # Next 3: rotation (axis-angle)
    qpos = hand_poses[:, 6:]  # Last 22: joint angles
    
    return hand_transl, hand_orient, qpos

def convert_dataset(input_file, output_dir):
    """Convert a single dataset from hand_poses format to qpos format."""
    print(f"Converting {input_file}...")
    
    # Load original data
    data = np.load(input_file, allow_pickle=True)
    
    # Extract components
    hand_poses = data['hand_poses']
    obj_pc = data['obj_pc']
    contact_labels = data['contact_labels']
    object_scales = data['object_scales'] 
    hand_sides = data['hand_sides']
    num_samples = data['num_samples']
    
    # Convert hand_poses to separate components
    hand_transl, hand_orient, qpos = convert_hand_poses_to_qpos_format(hand_poses)
    
    print(f"  Original hand_poses shape: {hand_poses.shape}")
    print(f"  Converted - hand_transl: {hand_transl.shape}, hand_orient: {hand_orient.shape}, qpos: {qpos.shape}")
    
    # Create object pose data (using identity for now, as we don't have object tracking)
    num_samples = hand_poses.shape[0]
    object_transl = np.zeros((num_samples, 3))  # Object at origin
    object_orient = np.tile(np.eye(3), (num_samples, 1, 1))  # Identity rotation
    
    # For object points and colors, we'll use the point cloud data
    # obj_pc is [N, num_points, 3], we need to handle this properly
    object_points = obj_pc  # [N, num_points, 3]
    object_colors = np.ones_like(obj_pc) * 0.5  # Default gray color
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each sample as a separate file (matching the working_format structure)
    for i in range(num_samples):
        output_file = os.path.join(output_dir, f"{i}.npz")
        
        # Create data dict for this sample
        sample_data = {
            'qpos': qpos[i:i+1],  # Keep batch dimension for consistency
            'hand_transl': hand_transl[i:i+1],
            'hand_orient': hand_orient[i:i+1], 
            'object_transl': object_transl[i:i+1],
            'object_orient': object_orient[i:i+1],
            'object_points': object_points[i],  # [num_points, 3]
            'object_colors': object_colors[i]   # [num_points, 3]
        }
        
        np.savez(output_file, **sample_data)
        
    print(f"  Saved {num_samples} samples to {output_dir}")
    return num_samples

def main():
    """Convert all datasets to qpos format."""
    
    # Input and output directories
    input_bases = [
        "data/your_data_converted",
        "data/your_data_converted_fixed"
    ]
    output_base = "data/qpos_format"
    
    total_samples = 0
    all_datasets = []
    
    # Process each input directory
    for input_base in input_bases:
        if not os.path.exists(input_base):
            print(f"Warning: {input_base} not found, skipping...")
            continue
            
        # Find all datasets in this directory
        dataset_dirs = [d for d in os.listdir(input_base) 
                       if os.path.isdir(os.path.join(input_base, d))]
        
        print(f"Found {len(dataset_dirs)} datasets in {input_base}: {dataset_dirs}")
        
        for dataset_name in dataset_dirs:
            # Handle both converted_data.npz and converted_data_fixed.npz
            possible_files = [
                os.path.join(input_base, dataset_name, "converted_data.npz"),
                os.path.join(input_base, dataset_name, "converted_data_fixed.npz")
            ]
            
            input_file = None
            for pf in possible_files:
                if os.path.exists(pf):
                    input_file = pf
                    break
            
            if input_file:
                # Create unique output directory based on source
                if "fixed" in input_base:
                    output_dir = os.path.join(output_base, f"{dataset_name}_fixed")
                else:
                    output_dir = os.path.join(output_base, dataset_name)
                
                samples = convert_dataset(input_file, output_dir)
                total_samples += samples
                all_datasets.append(f"{dataset_name} ({'fixed' if 'fixed' in input_base else 'original'})")
            else:
                print(f"Warning: No data file found for {dataset_name} in {input_base}, skipping...")
    
    print(f"\nConversion complete! Total samples: {total_samples}")
    print(f"Converted datasets: {all_datasets}")
    print(f"Output directory: {output_base}")

if __name__ == "__main__":
    main()