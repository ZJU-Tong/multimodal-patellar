import torch
import torch.nn.functional as F
import torch.nn as nn

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=1.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        weights = torch.ones_like(target)
        weights[target == 1] = self.pos_weight
        return F.binary_cross_entropy_with_logits(input, target.float(), weights)
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 平衡因子
        self.gamma = gamma  # 调制因子
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # 计算pt
        pt = torch.exp(-bce_loss)
        
        # 计算focal loss
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        return focal_loss.mean()