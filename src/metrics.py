# src/metrics.py
import torch

def dice_score(pred, target, smooth=1e-5):
    """
    Compute Dice Score between prediction and ground truth.
    Both should be tensors of shape (B, 1, D, H, W)
    """
    pred = torch.sigmoid(pred)  # Convert logits → probabilities
    pred = (pred > 0.5).float()  # Threshold to binary

    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(pred, target, smooth=1e-5):
    """
    Compute Intersection over Union (IoU).
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = (pred + target).clamp(0,1).sum(dim=(1,2,3,4))

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
