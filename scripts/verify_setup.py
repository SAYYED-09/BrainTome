#!/usr/bin/env python3
"""
BrainTome Setup Verification Script
This script checks if your environment is properly configured for BrainTome.
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('nibabel', 'NiBabel'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
        ('sklearn', 'scikit-learn')
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {name} {version} - OK")
        except ImportError:
            print(f"   ❌ {name} - Not installed")
            all_good = False
    
    return all_good

def check_pytorch_setup():
    """Check PyTorch configuration."""
    print("\n🔥 Checking PyTorch setup...")
    
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA version: {torch.version.cuda}")
        else:
            print("   ⚠️  CUDA not available - will use CPU training")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   ✅ MPS (Apple Silicon) available")
        
        return True
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'src',
        'preprocessing',
        'notebooks',
        'results',
        'data'
    ]
    
    required_files = [
        'src/model.py',
        'src/train.py',
        'src/inference.py',
        'src/dataset.py',
        'requirements.txt'
    ]
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ {dir_name}/ - OK")
        else:
            print(f"   ❌ {dir_name}/ - Missing")
            all_good = False
    
    # Check files
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"   ✅ {file_name} - OK")
        else:
            print(f"   ❌ {file_name} - Missing")
            all_good = False
    
    return all_good

def check_model_import():
    """Check if model can be imported."""
    print("\n🧠 Checking model import...")
    
    try:
        sys.path.append('src')
        from model import UNet3D
        
        # Try to create model
        model = UNet3D(in_channels=4, out_channels=1)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model created successfully")
        print(f"   ✅ Parameters: {param_count:,}")
        return True
    except Exception as e:
        print(f"   ❌ Model import failed: {e}")
        return False

def check_data_directory():
    """Check data directory setup."""
    print("\n💾 Checking data setup...")
    
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    
    if not data_dir.exists():
        print("   ⚠️  data/ directory not found - create it for dataset")
        return False
    
    if not processed_dir.exists():
        print("   ⚠️  data/processed/ not found - run preprocessing first")
        return False
    
    # Check for sample data
    sample_patients = list(processed_dir.glob('BraTS-GLI-*'))
    if sample_patients:
        print(f"   ✅ Found {len(sample_patients)} processed patients")
        return True
    else:
        print("   ⚠️  No processed patients found - run preprocessing")
        return False

def main():
    """Run all verification checks."""
    print("🧠 BrainTome Setup Verification")
    print("=" * 40)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_pytorch_setup(),
        check_project_structure(),
        check_model_import(),
        check_data_directory()
    ]
    
    print("\n" + "=" * 40)
    
    if all(checks[:5]):  # Exclude data check from critical checks
        print("🎉 Setup verification completed successfully!")
        print("   Your environment is ready for BrainTome.")
        
        if not checks[5]:
            print("\n📝 Next steps:")
            print("   1. Download BraTS 2025 dataset")
            print("   2. Run: python preprocessing/load_data.py")
            print("   3. Start training: python src/train.py")
    else:
        print("❌ Setup verification failed!")
        print("   Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()