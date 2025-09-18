#!/usr/bin/env python3
"""
BrainTome Quick Start Script
This script helps new users get started with BrainTome quickly.
"""

import os
import sys
import argparse
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw',
        'data/processed', 
        'results/models',
        'results/logs',
        'results/samples',
        'results/inference',
        'logs'
    ]
    
    print("📁 Creating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")

def download_sample_data():
    """Download sample data for testing."""
    print("📥 Sample data download not implemented yet.")
    print("   Please download BraTS 2025 dataset manually:")
    print("   https://www.synapse.org/#!Synapse:syn53708249")

def run_preprocessing(data_dir):
    """Run data preprocessing."""
    if not Path(data_dir).exists():
        print(f"❌ Data directory {data_dir} not found!")
        return False
    
    print("🔄 Running preprocessing...")
    cmd = f"python preprocessing/load_data.py --input_dir {data_dir} --output_dir data/processed"
    print(f"   Command: {cmd}")
    
    # Note: In a real implementation, you'd run this command
    print("   ⚠️  Please run the preprocessing command manually")
    return True

def run_quick_training():
    """Run a quick training session."""
    print("🚀 Starting quick training (1 epoch)...")
    cmd = "python src/train.py --data_dir data/processed --epochs 1 --batch_size 2"
    print(f"   Command: {cmd}")
    
    # Note: In a real implementation, you'd run this command
    print("   ⚠️  Please run the training command manually")

def main():
    parser = argparse.ArgumentParser(description="BrainTome Quick Start")
    parser.add_argument("--data-dir", type=str, help="Path to raw BraTS data")
    parser.add_argument("--skip-data", action="store_true", help="Skip data setup")
    parser.add_argument("--skip-training", action="store_true", help="Skip training demo")
    
    args = parser.parse_args()
    
    print("🧠 BrainTome Quick Start")
    print("=" * 30)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Data setup
    if not args.skip_data:
        if args.data_dir:
            run_preprocessing(args.data_dir)
        else:
            download_sample_data()
    
    # Step 3: Quick training
    if not args.skip_training:
        run_quick_training()
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Verify setup: python scripts/verify_setup.py")
    print("2. Explore notebooks: jupyter notebook notebooks/")
    print("3. Full training: python src/train.py --data_dir data/processed")
    print("4. Run inference: python src/inference.py --model_path results/best_model.pt")

if __name__ == "__main__":
    main()