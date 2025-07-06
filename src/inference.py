# src/inference.py

import os
import numpy as np
import nibabel as nib
import torch
from model import UNet3D

def load_patient_data(patient_id, processed_dir):
    modalities = ["t1n", "t1c", "t2w", "t2f"]
    volumes = []

    for mod in modalities:
        path = os.path.join(processed_dir, patient_id, f"{patient_id}-{mod}_resized.nii.gz")
        vol = nib.load(path).get_fdata()
        volumes.append(vol)

    img = np.stack(volumes, axis=0)  # Shape: (4, 128, 128, 128)
    return torch.tensor(img[np.newaxis], dtype=torch.float32)  # Add batch dim

def run_inference(model_path, patient_id, processed_dir, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = load_patient_data(patient_id, processed_dir).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        pred = (pred > 0.5).float().cpu().numpy()[0, 0]  # (128,128,128)

    # Save prediction as NIfTI
    seg_path = os.path.join(processed_dir, patient_id, f"{patient_id}-seg_resized.nii.gz")
    seg_ref = nib.load(seg_path)
    pred_nifti = nib.Nifti1Image(pred, affine=seg_ref.affine)
    os.makedirs(save_path, exist_ok=True)
    nib.save(pred_nifti, os.path.join(save_path, f"{patient_id}_pred.nii.gz"))
    print(f"[✅] Prediction saved to {os.path.join(save_path, f'{patient_id}_pred.nii.gz')}")

if __name__ == "__main__":
    patient_id = "BraTS-GLI-00025-000"  # ← Change if needed
    processed_dir = "data/processed"
    model_path = "results/best_model.pt"
    save_path = "results/inference"
    run_inference(model_path, patient_id, processed_dir, save_path)
