# src/train.py

import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import PatchDataset
from model import UNet3D
from metrics import dice_score
from utils import save_prediction_slice, count_trainable_params

def train_model(data_dir, epochs=5, batch_size=2, lr=1e-4, save_dir="results"):
    # Load and subset dataset
    full_dataset = PatchDataset(data_dir)
    subset_size = min(1000, len(full_dataset))
    dataset = Subset(full_dataset, range(subset_size))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")

    # Model setup
    model = UNet3D().to(device)
    count_trainable_params(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    best_dice = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        dice_total = 0.0

        for batch_idx, (x, y) in enumerate(tqdm(loader, desc=f"[Epoch {epoch+1}/{epochs}]")):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            dice_total += dice_score(outputs, y)

            # Save a visual slice every 1st batch of every 2nd epoch
            if batch_idx == 0 and epoch % 2 == 0:
                save_prediction_slice(outputs[0], y[0], os.path.join(save_dir, "samples"), epoch, batch_idx)

        avg_loss = epoch_loss / len(loader)
        avg_dice = dice_total / len(loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Dice Score: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[✔] Best model updated. Dice = {best_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="results")

    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir)
