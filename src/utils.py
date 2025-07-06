# src/utils.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_prediction_slice(pred, target, path, epoch=None, batch_idx=None):
    """
    Save side-by-side prediction and ground truth for a middle slice.
    pred, target: tensors of shape (1, D, H, W)
    """
    os.makedirs(path, exist_ok=True)

    pred_np = torch.sigmoid(pred).detach().cpu().numpy()[0]
    target_np = target.detach().cpu().numpy()[0]

    # Pick middle slice
    mid_slice = pred_np.shape[0] // 2
    pred_slice = pred_np[mid_slice]
    target_slice = target_np[mid_slice]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(pred_slice, cmap="nipy_spectral")
    axs[0].set_title("Prediction")
    axs[0].axis("off")

    axs[1].imshow(target_slice, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    tag = f"epoch{epoch}_batch{batch_idx}" if epoch is not None else "sample"
    save_path = os.path.join(path, f"{tag}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def count_trainable_params(model):
    """
    Print total trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable Parameters: {total:,}")
