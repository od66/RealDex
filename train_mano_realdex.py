#!/usr/bin/env python3
"""
Official RealDex MANO Training Script (PyTorch3D-Free)
Uses the existing RealDex MANO implementation without PyTorch3D dependencies
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm

# Add paths for RealDex modules
sys.path.append('dexgrasp_generation')
sys.path.append('dexgrasp_generation/network')
sys.path.append('dexgrasp_generation/utils')

# Import RealDex MANO implementation
from dexgrasp_generation.utils.grab_hand_model import HandModel
from dexgrasp_generation.network.models.loss import contact_map_of_m_to_n

class ContactMapNetwork(nn.Module):
    """Contact Map Network for MANO hand"""
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # obj_pc + hand_pc
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data_dict):
        obj_pc = data_dict['canon_obj_pc']  # [B, N, 3]
        hand_pc = data_dict['observed_hand_pc']  # [B, M, 3]
        
        # Simple concatenation approach
        B = obj_pc.shape[0]
        obj_features = obj_pc.view(B, -1)  # [B, N*3]
        hand_features = hand_pc.view(B, -1)  # [B, M*3]
        
        # Pad to same size
        max_size = max(obj_features.shape[1], hand_features.shape[1])
        if obj_features.shape[1] < max_size:
            obj_features = torch.cat([obj_features, torch.zeros(B, max_size - obj_features.shape[1], device=obj_features.device)], dim=1)
        if hand_features.shape[1] < max_size:
            hand_features = torch.cat([hand_features, torch.zeros(B, max_size - hand_features.shape[1], device=hand_features.device)], dim=1)
            
        combined = torch.cat([obj_features, hand_features], dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        contact_map = self.fc3(x)  # [B, output_dim]
        
        # Reshape to match expected output
        N = obj_pc.shape[1]
        contact_map = contact_map.unsqueeze(1).expand(-1, N, -1)  # [B, N, output_dim]
        
        return {'contact_map': contact_map}

class RealDexMANODataset(Dataset):
    """RealDex dataset for MANO training without PyTorch3D dependencies"""
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.data = self._load_data()
        
    def _load_data(self):
        # Simple synthetic data for testing
        # In practice, this would load from RealDex dataset files
        data = {
            'qpos': torch.randn(100, 48),  # MANO pose parameters
            'hand_transl': torch.randn(100, 3),
            'hand_orient': torch.randn(100, 3),
            'object_points': [torch.randn(1024, 3) for _ in range(100)],
            'object_names': ['test_object'] * 100
        }
        return data
        
    def __len__(self):
        return len(self.data['qpos'])
        
    def __getitem__(self, idx):
        return {
            'obj_pc': self.data['object_points'][idx],
            'hand_qpos': self.data['qpos'][idx],
            'translation': self.data['hand_transl'][idx],
            'rotation': self.data['hand_orient'][idx],
            'object_name': self.data['object_names'][idx]
        }

def train_mano_contact_network():
    """Train contact map network using RealDex MANO implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize MANO hand model
    print("Initializing MANO hand model...")
    hand_model = HandModel(device=device, n_surface_points=1024)
    
    # Initialize contact map network
    print("Initializing contact map network...")
    contact_net = ContactMapNetwork().to(device)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = RealDexMANODataset('data/realdex_data', mode='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(contact_net.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting training...")
    contact_net.train()
    
    for epoch in range(5):  # Small number for testing
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Move to device
            obj_pc = batch['obj_pc'].to(device)
            hand_qpos = batch['hand_qpos'].to(device)
            translation = batch['translation'].to(device)
            rotation = batch['rotation'].to(device)
            
            # Create hand pose tensor
            hand_pose = torch.cat([translation, rotation, hand_qpos], dim=-1)
            
            # Forward pass through MANO hand model
            hand_output = hand_model(hand_pose, obj_pc, 
                                   with_penetration=True, 
                                   with_surface_points=True, 
                                   with_contact_candidates=True)
            
            # Get hand surface points
            hand_surface_points = hand_output['surface_points']
            
            # Forward pass through contact network
            contact_pred = contact_net({
                'canon_obj_pc': obj_pc,
                'observed_hand_pc': hand_surface_points
            })
            
            # Compute ground truth contact map
            contact_gt = hand_output['cmap']  # From MANO hand model
            
            # Compute loss
            loss = criterion(contact_pred['contact_map'].squeeze(-1), contact_gt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the trained model
    model_path = 'official_mano_contact_model.pth'
    torch.save({
        'contact_net_state_dict': contact_net.state_dict(),
        'hand_model_state_dict': hand_model.state_dict(),
        'epoch': epoch + 1,
        'loss': avg_loss
    }, model_path)
    
    print(f"✅ Training completed! Model saved to {model_path}")
    print("🎯 Successfully used official RealDex MANO implementation!")

if __name__ == "__main__":
    # Set MANO model path
    os.environ['MANO_MODELS_PATH'] = '/home/od66/GRILL/RealDex/assets/mano/models'
    
    print("🚀 Starting Official RealDex MANO Training")
    print("📁 MANO models path:", os.environ.get('MANO_MODELS_PATH'))
    print("🤖 Using official RealDex MANO implementation from grab_hand_model.py")
    
    train_mano_contact_network()