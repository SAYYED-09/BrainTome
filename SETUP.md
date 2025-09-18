# 🛠️ BrainTome Setup Guide

This guide provides detailed instructions for setting up BrainTome on different systems.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space for dataset
- **GPU**: Optional (CUDA-compatible for faster training)

### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3060 or better with 8GB+ VRAM
- **Storage**: SSD with 100GB+ free space

## 🚀 Installation Methods

### Method 1: Standard Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/BrainTome.git
cd BrainTome
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv braintome_env
source braintome_env/bin/activate  # Linux/Mac
# braintome_env\Scripts\activate  # Windows

# Using conda (alternative)
conda create -n braintome python=3.9
conda activate braintome
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Method 2: Development Installation

```bash
git clone https://github.com/yourusername/BrainTome.git
cd BrainTome
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## 📊 Dataset Setup

### Option 1: BraTS 2025 Dataset (Recommended)

1. **Register and Download**
   - Visit [BraTS 2025 Challenge](https://www.synapse.org/#!Synapse:syn53708249)
   - Create account and accept terms
   - Download training dataset

2. **Extract Dataset**
```bash
# Create data directory
mkdir -p data/raw

# Extract downloaded files
unzip BraTS2025_Training_Data.zip -d data/raw/
```

3. **Preprocess Data**
```bash
python preprocessing/load_data.py \
    --input_dir data/raw/BraTS2025_Training_Data \
    --output_dir data/processed \
    --resize_to 128 128 128
```

### Option 2: Sample Dataset (Quick Start)

```bash
# Download sample data (smaller subset for testing)
python scripts/download_sample_data.py --output_dir data/sample
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Data paths
DATA_DIR=data/processed
MODEL_DIR=results
LOG_DIR=logs

# Training settings
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_EPOCHS=10

# Hardware settings
DEVICE=cuda  # or 'cpu'
NUM_WORKERS=4
```

### Config File
Edit `config/config.yaml`:

```yaml
data:
  input_dir: "data/processed"
  batch_size: 4
  num_workers: 4

model:
  in_channels: 4
  out_channels: 1
  base_filters: 32

training:
  epochs: 10
  learning_rate: 1e-4
  optimizer: "adam"
  scheduler: "cosine"

inference:
  threshold: 0.5
  tta: false  # Test Time Augmentation
```

## ✅ Verification

### Test Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import nibabel; print('NiBabel: OK')"
python -c "from src.model import UNet3D; print('Model import: OK')"
```

### Run Quick Test
```bash
# Test model creation
python scripts/test_model.py

# Test data loading (requires dataset)
python scripts/test_data_loading.py

# Run minimal training (1 epoch)
python src/train.py --data_dir data/processed --epochs 1 --batch_size 1
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python src/train.py --batch_size 1

# Use CPU training
python src/train.py --device cpu
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# For conda users
conda install pytorch torchvision -c pytorch
```

#### Data Loading Errors
```bash
# Check data directory structure
python scripts/verify_data_structure.py --data_dir data/processed

# Rerun preprocessing
python preprocessing/load_data.py --input_dir data/raw --output_dir data/processed
```

### Platform-Specific Issues

#### Windows
- Use PowerShell or Command Prompt as Administrator
- Install Visual Studio Build Tools if compilation errors occur
- Use forward slashes in paths or raw strings

#### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- For M1/M2 Macs, use MPS backend: `--device mps`

#### Linux
- Install system dependencies: `sudo apt-get install python3-dev`
- For CUDA: Install NVIDIA drivers and CUDA toolkit

## 🚀 Performance Optimization

### GPU Setup
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-specific PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Optimization
```bash
# Enable memory efficient training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use gradient checkpointing
python src/train.py --gradient_checkpointing
```

## 📚 Next Steps

After successful setup:

1. **Explore Notebooks**: Start with `notebooks/01_load_and_visualize_BraTS2025.ipynb`
2. **Run Training**: `python src/train.py --data_dir data/processed`
3. **Test Inference**: `python src/inference.py --model_path results/best_model.pt`
4. **Visualize Results**: Use notebooks in `notebooks/` directory

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Search existing [GitHub Issues](https://github.com/yourusername/BrainTome/issues)
3. Create a new issue with:
   - System information
   - Error messages
   - Steps to reproduce
   - Expected vs actual behavior

## 📞 Support

- **Documentation**: [Project Wiki](https://github.com/yourusername/BrainTome/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/BrainTome/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/BrainTome/discussions)

Happy coding! 🧠✨