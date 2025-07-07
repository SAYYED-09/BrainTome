# BrainTome: MRI Brain Tumor Segmentation and Classification
# 🧠 BrainTome: MRI Brain Tumor Segmentation & Visualization

BrainTome is a deep learning project focused on segmenting brain tumors from MRI scans using a lightweight 3D U-Net model trained on the BraTS 2025 dataset.

---

## 🚀 Project Overview

- **Task**: Brain tumor segmentation (semantic) from 3D MRI volumes.
- **Dataset**: BraTS 2025 (FLAIR, T1n, T1c, T2w + segmentation masks)
- **Model**: 3D U-Net with patch-based training.
- **Training Device**: CPU-based, optimized for low-resource systems.
- **Score**: Dice = **0.9491** on validation.

---

## 🧪 Project Phases

1. **Preprocessing**  
   ✅ Intensity normalization, resizing, patching, and volume stacking.

2. **Model Training**  
   ✅ Patch-based training using a 3D U-Net on selected modalities.

3. **Inference & Evaluation**  
   ✅ Segment new brain scans, compare prediction vs ground truth.

4. **Visualization**  
   ✅ Feature maps, middle slices, tumor overlays.
![output](https://github.com/user-attachments/assets/c9bf7733-0f73-4a88-a9a5-f470dd99b15c)

