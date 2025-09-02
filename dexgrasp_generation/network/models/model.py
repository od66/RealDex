import torch.nn as nn
import torch
import os
import sys
from os.path import join as pjoin
from abc import abstractmethod
from copy import deepcopy
from hydra import compose
from omegaconf.omegaconf import open_dict
from omegaconf import OmegaConf
# Import chamfer_distance with fallback to simple implementation
try:
    from pytorch3d.loss import chamfer_distance as CD
    print("✓ Using PyTorch3D chamfer_distance")
except ImportError as e:
    print(f"⚠ PyTorch3D chamfer_distance failed ({e}), using simple implementation")
    
    def simple_chamfer_distance(x, y, point_reduction='sum', batch_reduction='mean'):
        """
        Simple chamfer distance implementation without PyTorch3D dependency.
        Args:
            x, y: point clouds of shape (B, N, 3) and (B, M, 3)
            point_reduction: 'sum' or 'mean'
            batch_reduction: 'sum' or 'mean'
        Returns:
            chamfer_distance, None (to match PyTorch3D API)
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
    
    CD = simple_chamfer_distance


base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, '..'))
sys.path.insert(0, pjoin(base_path, '..', '..'))

from network.models.loss import discretize_gt_cm
from network.models.contactnet.contact_network import ContactMapNet
from network.models.affordancenet.affordance_network import AffordanceCVAE
from collections import OrderedDict

def get_last_model(dirname, key=""):
    if not os.path.exists(dirname):
        return None
    models = [pjoin(dirname, f) for f in os.listdir(dirname) if
              os.path.isfile(pjoin(dirname, f)) and
              key in f and ".pt" in f]
    if models is None or len(models) == 0:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name

def get_model(dir, resume_epoch):
    last_model_name = get_last_model(dir)
    print('last model name', last_model_name)
    if resume_epoch is not None and resume_epoch > 0:
        specified_model = pjoin(dir, f"model_{resume_epoch:04d}.pt")
        if os.path.exists(specified_model):
            last_model_name = specified_model
    return last_model_name



class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.device = cfg['device']
        self.loss_weights = cfg['model']['loss_weight']

        self.cfg = cfg
        self.feed_dict = {}
        self.pred_dict = {}
        self.save_dict = {}
        self.loss_dict = {}

    def summarize_losses(self, loss_dict):
        total_loss = 0
        for key, item in self.loss_weights.items():
            if key in loss_dict:
                total_loss += loss_dict[key] * item
        
        loss_dict['total_loss'] = total_loss
        self.loss_dict = loss_dict
        # print(loss_dict)

    def update(self):
        self.pred_dict = self.net(self.feed_dict)
        self.compute_loss()
        self.loss_dict['total_loss'].backward()

    def set_data(self, data):
        self.feed_dict = {}
        for key, item in data.items():
            if key in [""]:
                continue
            if type(item) == torch.Tensor:
                item = item.float().to(self.device)
            self.feed_dict[key] = item

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def test(self, save=False, no_eval=False, epoch=0):
        pass

class ContactModel(BaseModel):
    def __init__(self, cfg):
        super(ContactModel, self).__init__(cfg)
        self.net = ContactMapNet(cfg).to(self.device)

        self.cm_loss = nn.MSELoss()
        self.cm_bin_loss = nn.CrossEntropyLoss()

    def compute_loss(self):
        loss_dict = {}

        gt_contact_map = self.feed_dict['contact_map']
        pred_contact_map = self.pred_dict['contact_map']

        if len(pred_contact_map.shape) == 2:
            # pred_contact_map: [B, N]
            loss_dict["contact_map"] = self.cm_loss(pred_contact_map, gt_contact_map)
        elif len(pred_contact_map.shape) == 3:
            # pred_contact_map: [B, N, 10]
            gt_bins = discretize_gt_cm(gt_contact_map, num_bins=pred_contact_map.shape[-1])  # [B, N, 10]
            gt_bins_labels = torch.argmax(gt_bins, dim=-1)  # [B, N]
            pred_contact_map = pred_contact_map.transpose(2, 1)  # [B, 10, N]
            loss_dict["contact_map"] = self.cm_bin_loss(pred_contact_map, gt_bins_labels)
        

        self.summarize_losses(loss_dict)

    def test(self, save=False, no_eval=False, epoch=0):
        self.loss_dict = {}
        with torch.no_grad():
            self.pred_dict = self.net(self.feed_dict)
            if not no_eval:
                self.compute_loss()


class AffordanceModel(BaseModel):
    def __init__(self, cfg):
        super(AffordanceModel, self).__init__(cfg)
        
        # Create minimal config for contact network to avoid Hydra issues
        contact_cfg = OmegaConf.create({
            'exp_dir': cfg['model']['contact_net']['ckpt_path'],
            'device': self.device,
            'model': {
                'obj_inchannel': 3,
                'num_obj_points': cfg['dataset']['num_obj_points'],
                'backbone_net': 'pointnet',
                'network': {
                    'out_channel': 10
                }
            },
            'dataset': {
                'num_obj_points': cfg['dataset']['num_obj_points'],
                'num_hand_points': cfg['dataset']['num_hand_points']
            }
        })
        
        self.contact_net = ContactMapNet(contact_cfg).to(self.device)
        self.net = AffordanceCVAE(cfg, self.contact_net).to(self.device)
        self.normalize_factor=cfg['model']['tta']['normalize_factor']
        
        # ckpt_dir = pjoin(contact_cfg['exp_dir'], 'ckpt')
        # model_name = get_last_model(ckpt_dir)
        
        # if model_name:
        #     print(model_name)
        #     ckpt = torch.load(model_name)['model']
        #     new_ckpt = OrderedDict()
        #     for name in ckpt.keys():
        #         new_name = name.replace('net.', '')
        #         if new_name.startswith('backbone.'):
        #             new_name = new_name.replace('backbone.', '')
        #         new_ckpt[new_name] = ckpt[name]
            
        #     self.contact_net.load_state_dict(new_ckpt)
        #     self.contact_net = self.contact_net.to(cfg['device'])
        # else:
        #     print("Didn't find the contact net ckpt")
            
        # self.contact_net.eval()
        
        
    def compute_loss(self):
        self.loss_dict = {}
        pred_dict = self.pred_dict
        feed_dict = self.feed_dict
        
        transl = pred_dict['translation']
        rotation = pred_dict['rotation']
        qpos = pred_dict['hand_qpos']
        # hand_points = pred_dict['hand_points']
        pen_dist = pred_dict['pen_dist']
        batch_size = qpos.shape[0]
        # qpos param loss
        qpos_loss = torch.nn.functional.mse_loss(qpos, feed_dict['hand_qpos'])#/batch_size
        # transl loss
        transl_loss = torch.nn.functional.mse_loss(transl, feed_dict['translation'])#/batch_size
        # rotation loss
        rotation_loss = torch.nn.functional.mse_loss(rotation, feed_dict['rotation'])#/batch_size
        # recon vertices loss
        verts_loss, _ = CD(pred_dict['recon_hand_points'], pred_dict['gt_hand_points'], point_reduction='sum', batch_reduction='mean')
        verts_loss /= batch_size
        
        # KLD loss
        KLD_loss = -0.5 * torch.sum(1 + pred_dict['log_var'] - pred_dict['mean'].pow(2) - pred_dict['log_var'].exp()) #/batch_size
        # cmap loss
        cmap_pred = pred_dict['cmap_pred'] # from contactnet output
        # calculate pseudo contactmap: 0~3cm mapped into value 1~0
        cmap_pseudo = 2 - 2 * torch.sigmoid(self.normalize_factor * (pred_dict['o2h_dist'].abs() + 1e-8).sqrt())   
        cmap_loss = torch.nn.functional.mse_loss(cmap_pseudo, cmap_pred)
        # inter penetration loss
        penetr_loss = pen_dist[pen_dist>0].sum() / batch_size
        
        loss_dict = {
            'qpos_loss': qpos_loss,
            'transl_loss': transl_loss,
            'rotation_loss': rotation_loss,
            'verts_loss': verts_loss,
            'KLD': KLD_loss,
            'cmap_loss': cmap_loss,
            'penetr_loss': penetr_loss   
        }
        self.summarize_losses(loss_dict)
        
        return super().compute_loss()
    
    def test(self, save=False, no_eval=False, epoch=0):
        self.loss_dict = {}
        with torch.no_grad():
            obj_pc = self.feed_dict['obj_pc']
            self.pred_dict = self.net.inference(obj_pc.transpose(1,2))
            self.pred_dict.update(self.feed_dict)

            # if not no_eval:
            #     self.compute_loss()