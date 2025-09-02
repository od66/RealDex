"""
PyTorch3D Compatibility Module for RealDex
Provides fallback implementations for PyTorch3D functions that have ABI compatibility issues.
"""

import torch
import numpy as np

# Try to import PyTorch3D functions, with fallbacks for problematic ones
try:
    from pytorch3d.structures import Meshes
    print("✓ PyTorch3D Meshes available")
except ImportError:
    print("✗ PyTorch3D Meshes not available")
    Meshes = None

try:
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    print("✓ PyTorch3D transforms available")
except ImportError:
    print("⚠ PyTorch3D transforms not available, using fallback implementations")
    
    def axis_angle_to_matrix(axis_angle):
        """Convert axis-angle to rotation matrix."""
        # Simple implementation using Rodrigues' rotation formula
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / (angle + 1e-8)
        
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
        
        # Rodrigues' formula
        rotation_matrix = torch.zeros(*axis_angle.shape[:-1], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
        
        rotation_matrix[..., 0, 0] = cos_angle.squeeze(-1) + x.squeeze(-1)**2 * one_minus_cos.squeeze(-1)
        rotation_matrix[..., 0, 1] = x.squeeze(-1) * y.squeeze(-1) * one_minus_cos.squeeze(-1) - z.squeeze(-1) * sin_angle.squeeze(-1)
        rotation_matrix[..., 0, 2] = x.squeeze(-1) * z.squeeze(-1) * one_minus_cos.squeeze(-1) + y.squeeze(-1) * sin_angle.squeeze(-1)
        
        rotation_matrix[..., 1, 0] = y.squeeze(-1) * x.squeeze(-1) * one_minus_cos.squeeze(-1) + z.squeeze(-1) * sin_angle.squeeze(-1)
        rotation_matrix[..., 1, 1] = cos_angle.squeeze(-1) + y.squeeze(-1)**2 * one_minus_cos.squeeze(-1)
        rotation_matrix[..., 1, 2] = y.squeeze(-1) * z.squeeze(-1) * one_minus_cos.squeeze(-1) - x.squeeze(-1) * sin_angle.squeeze(-1)
        
        rotation_matrix[..., 2, 0] = z.squeeze(-1) * x.squeeze(-1) * one_minus_cos.squeeze(-1) - y.squeeze(-1) * sin_angle.squeeze(-1)
        rotation_matrix[..., 2, 1] = z.squeeze(-1) * y.squeeze(-1) * one_minus_cos.squeeze(-1) + x.squeeze(-1) * sin_angle.squeeze(-1)
        rotation_matrix[..., 2, 2] = cos_angle.squeeze(-1) + z.squeeze(-1)**2 * one_minus_cos.squeeze(-1)
        
        return rotation_matrix
    
    def matrix_to_axis_angle(rotation_matrix):
        """Convert rotation matrix to axis-angle representation."""
        # Extract rotation angle
        trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        # Extract rotation axis
        axis = torch.stack([
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]
        ], dim=-1)
        
        # Normalize axis
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)
        
        # Scale by angle
        axis_angle = axis * angle.unsqueeze(-1)
        
        return axis_angle

# Problematic functions - provide fallbacks
try:
    from pytorch3d.ops.knn import knn_points, knn_gather
    from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
    print("✓ PyTorch3D ops available")
except ImportError as e:
    print(f"⚠ PyTorch3D ops failed ({e}), using fallback implementations")
    
    def knn_points(p1, p2, K=1):
        """
        Simple KNN implementation without PyTorch3D dependency.
        Args:
            p1: (B, N, 3) query points
            p2: (B, M, 3) reference points  
            K: number of nearest neighbors
        Returns:
            namedtuple with .dists (B, N, K) and .idx (B, N, K)
        """
        from collections import namedtuple
        
        # Compute pairwise distances
        p1_expanded = p1.unsqueeze(2)  # (B, N, 1, 3)
        p2_expanded = p2.unsqueeze(1)  # (B, 1, M, 3)
        distances = torch.sum((p1_expanded - p2_expanded) ** 2, dim=3)  # (B, N, M)
        
        # Find K nearest neighbors
        dists, idx = torch.topk(distances, K, dim=2, largest=False)  # (B, N, K)
        
        KNNResult = namedtuple('KNNResult', ['dists', 'idx'])
        return KNNResult(dists=dists, idx=idx)
    
    def knn_gather(points, idx):
        """Gather points using KNN indices."""
        batch_size, num_points, num_dims = points.shape
        _, num_query, k = idx.shape
        
        # Expand points for gathering
        points_expanded = points.unsqueeze(1).expand(-1, num_query, -1, -1)  # (B, num_query, num_points, 3)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, num_dims)  # (B, num_query, k, 3)
        
        gathered = torch.gather(points_expanded, 2, idx_expanded)  # (B, num_query, k, 3)
        return gathered
    
    def sample_points_from_meshes(meshes, num_samples=1000):
        """
        Simple mesh sampling implementation.
        This is a basic implementation - in practice, you'd want more sophisticated sampling.
        """
        if meshes is None:
            raise ValueError("Meshes is None - PyTorch3D not available")
        
        # For now, return random points from mesh vertices as a fallback
        # This is not ideal but allows the code to run
        verts = meshes.verts_list()[0]  # Get first mesh vertices
        if len(verts) < num_samples:
            # If not enough vertices, repeat some
            indices = torch.randint(0, len(verts), (num_samples,))
        else:
            # Random sampling
            indices = torch.randperm(len(verts))[:num_samples]
        
        sampled_points = verts[indices].unsqueeze(0)  # Add batch dimension
        return sampled_points
    
    def sample_farthest_points(points, K=1000):
        """
        Simple farthest point sampling implementation.
        Args:
            points: (B, N, 3) point cloud
            K: number of points to sample
        Returns:
            tuple of (sampled_points, indices)
        """
        B, N, _ = points.shape
        if K >= N:
            return points, torch.arange(N).unsqueeze(0).repeat(B, 1)
        
        # Simple random sampling as fallback (not optimal but functional)
        indices = torch.randperm(N)[:K].unsqueeze(0).repeat(B, 1)
        sampled_points = torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        
        return sampled_points, indices

try:
    from pytorch3d.loss import chamfer_distance
    print("✓ PyTorch3D chamfer_distance available")
except ImportError as e:
    print(f"⚠ PyTorch3D chamfer_distance failed ({e}), using fallback implementation")
    
    def chamfer_distance(x, y, point_reduction='sum', batch_reduction='mean'):
        """
        Simple chamfer distance implementation without PyTorch3D dependency.
        """
        # x: (B, N, 3), y: (B, M, 3)
        x_expanded = x.unsqueeze(2)  # (B, N, 1, 3)
        y_expanded = y.unsqueeze(1)  # (B, 1, M, 3)
        
        # Compute squared distances
        distances = torch.sum((x_expanded - y_expanded) ** 2, dim=3)  # (B, N, M)
        
        # Forward chamfer: for each point in x, find closest in y
        min_dist_x_to_y = torch.min(distances, dim=2)[0]  # (B, N)
        
        # Backward chamfer: for each point in y, find closest in x
        min_dist_y_to_x = torch.min(distances, dim=1)[0]  # (B, M)
        
        # Combine forward and backward
        if point_reduction == 'sum':
            forward_chamfer = torch.sum(min_dist_x_to_y, dim=1)  # (B,)
            backward_chamfer = torch.sum(min_dist_y_to_x, dim=1)  # (B,)
        else:  # mean
            forward_chamfer = torch.mean(min_dist_x_to_y, dim=1)  # (B,)
            backward_chamfer = torch.mean(min_dist_y_to_x, dim=1)  # (B,)
        
        chamfer_dist = forward_chamfer + backward_chamfer  # (B,)
        
        if batch_reduction == 'mean':
            chamfer_dist = torch.mean(chamfer_dist)
        elif batch_reduction == 'sum':
            chamfer_dist = torch.sum(chamfer_dist)
        
        return chamfer_dist, None

print("✓ PyTorch3D compatibility module loaded")