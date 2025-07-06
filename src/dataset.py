# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class PatchDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # ✅ Only include directories (e.g., BraTS-GLI-00000-000)
        self.patient_ids = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.X_paths = []
        self.Y_paths = []

        for pid in self.patient_ids:
            folder = os.path.join(data_dir, pid)
            files = sorted([f for f in os.listdir(folder) if f.startswith("X_")])
            for file in files:
                idx = file.split("_")[1].split(".")[0]
                self.X_paths.append(os.path.join(folder, f"X_{idx}.npy"))
                self.Y_paths.append(os.path.join(folder, f"Y_{idx}.npy"))

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        x = np.load(self.X_paths[idx])       # Shape: (4, 64, 64, 64)
        y = np.load(self.Y_paths[idx])       # Shape: (1, 64, 64, 64)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
