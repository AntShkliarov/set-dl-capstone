"""
Dataset loader and preprocessor for drone audio classification
Separate from the classifier to allow flexible dataset manipulation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datasets import load_dataset, Dataset
import pickle
from tqdm import tqdm

class DroneDatasetLoader:
    """
    Independent dataset loader for drone audio classification
    Handles caching, sampling, and preprocessing
    """
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_cache_path = self.cache_dir / "drone_dataset.pkl"
        
    def load_raw_dataset(self, force_reload: bool = False):
        """Load raw dataset from HuggingFace"""
        if not force_reload and self.dataset_cache_path.exists():
            print("📂 Loading cached dataset...")
            with open(self.dataset_cache_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"✅ Cached dataset loaded: {len(dataset['train'])} samples")
            return dataset
        
        print("📥 Loading drone audio dataset from HuggingFace...")
        dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
        
        # Cache the dataset
        print("💾 Caching dataset for future use...")
        with open(self.dataset_cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"✅ Dataset loaded and cached: {len(dataset['train'])} samples")
        return dataset
    
    def sample_dataset(self, dataset, fraction: float = 1.0, random_seed: int = 42):
        """Sample a fraction of the dataset"""
        if fraction >= 1.0:
            return dataset
        
        original_size = len(dataset['train'])
        new_size = int(original_size * fraction)
        
        print(f"🔽 Sampling {fraction:.1%} of dataset: {original_size} → {new_size} samples")
        
        # Use shuffle and select for random sampling
        shuffled_dataset = dataset['train'].shuffle(seed=random_seed)
        sampled_dataset = shuffled_dataset.select(range(new_size))
        
        return {"train": sampled_dataset}
    
    def get_dataset_info(self, dataset):
        """Get comprehensive dataset information"""
        train_data = dataset['train']
        
        info = {
            'total_samples': len(train_data),
            'features': list(train_data.features.keys()),
            'audio_feature': train_data.features['audio'],
            'label_feature': train_data.features['label'],
        }
        
        # Sample analysis
        sample = train_data[0]
        audio_data = sample['audio']
        
        info.update({
            'sample_rate': audio_data['sampling_rate'],
            'sample_duration': len(audio_data['array']) / audio_data['sampling_rate'],
            'sample_length': len(audio_data['array']),
            'label_names': train_data.features['label'].names if hasattr(train_data.features['label'], 'names') else None,
        })
        
        # Label distribution
        labels = [sample['label'] for sample in train_data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        info['label_distribution'] = dict(zip(unique_labels, counts))
        
        return info
    
    def print_dataset_info(self, dataset):
        """Print formatted dataset information"""
        info = self.get_dataset_info(dataset)
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"📊 Total samples: {info['total_samples']:,}")
        print(f"🎵 Sample rate: {info['sample_rate']} Hz")
        print(f"⏱️  Sample duration: {info['sample_duration']:.2f} seconds")
        print(f"📏 Sample length: {info['sample_length']} points")
        
        if info['label_names']:
            print(f"🏷️  Labels: {info['label_names']}")
        
        print(f"📈 Label distribution:")
        for label, count in info['label_distribution'].items():
            percentage = (count / info['total_samples']) * 100
            label_name = info['label_names'][label] if info['label_names'] else f"Label {label}"
            print(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        print("="*50)
    
    def prepare_dataset_splits(self, dataset, test_size: float = 0.2, val_size: float = 0.2, random_seed: int = 42):
        """Create train/validation/test splits"""
        from sklearn.model_selection import train_test_split
        
        # Get all indices
        indices = list(range(len(dataset['train'])))
        labels = [dataset['train'][i]['label'] for i in indices]
        
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_seed, stratify=labels
        )
        
        # Second split: train vs val
        train_val_labels = [labels[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=random_seed, stratify=train_val_labels
        )
        
        # Create split datasets
        splits = {
            'train': dataset['train'].select(train_indices),
            'validation': dataset['train'].select(val_indices),
            'test': dataset['train'].select(test_indices)
        }
        
        print(f"📊 Dataset splits created:")
        print(f"   Train: {len(splits['train']):,} samples")
        print(f"   Validation: {len(splits['validation']):,} samples")
        print(f"   Test: {len(splits['test']):,} samples")
        
        return splits
    
    def save_processed_dataset(self, dataset, name: str):
        """Save processed dataset"""
        save_path = self.cache_dir / f"{name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"💾 Saved processed dataset: {save_path}")
        return save_path
    
    def load_processed_dataset(self, name: str):
        """Load processed dataset"""
        load_path = self.cache_dir / f"{name}.pkl"
        if load_path.exists():
            with open(load_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"📂 Loaded processed dataset: {load_path}")
            return dataset
        else:
            print(f"❌ Processed dataset not found: {load_path}")
            return None
    
    def load_processed_dataset_safe(self, dataset_name: str):
        """Safely load processed dataset with proper error handling"""
        print(f"📂 Loading processed dataset: '{dataset_name}'...")
        
        dataset = self.load_processed_dataset(dataset_name)
        if dataset is None:
            raise ValueError(
                f"Processed dataset '{dataset_name}' not found. "
                f"Available datasets might be found in {self.cache_dir}"
            )
        
        print(f"✅ Processed dataset '{dataset_name}' loaded successfully")
        return self._normalize_dataset_structure(dataset)
    
    def load_raw_dataset_safe(self, force_reload: bool = False):
        """Safely load raw dataset with error handling"""
        print("📥 Loading raw dataset...")
        
        try:
            dataset = self.load_raw_dataset(force_reload)
            if not dataset:
                raise ValueError("Raw dataset loading returned empty result")
            
            print("✅ Raw dataset loaded successfully")
            return self._normalize_dataset_structure(dataset)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load raw dataset: {str(e)}") from e
    
    def _normalize_dataset_structure(self, dataset):
        """Normalize dataset structure to ensure consistent dict format"""
        if dataset is None:
            raise ValueError("Dataset is None, cannot normalize structure")
        
        # If dataset is already a dict with multiple splits, return as-is
        if isinstance(dataset, dict) and len(dataset) > 1:
            print(f"📊 Dataset structure normalized: {list(dataset.keys())} splits available")
            return dataset
        
        # If dataset is a dict with only 'train', keep it
        if isinstance(dataset, dict) and 'train' in dataset:
            print("📊 Dataset structure normalized: single 'train' split")
            return dataset
        
        # If dataset is not a dict, assume it's a single Dataset object and wrap it
        if not isinstance(dataset, dict):
            print("📊 Dataset structure normalized: wrapped single dataset as 'train' split")
            return {'train': dataset}
        
        # Fallback - if we have a dict but no 'train' key, use the first available split
        if isinstance(dataset, dict) and 'train' not in dataset:
            first_key = next(iter(dataset.keys()))
            print(f"📊 Dataset structure normalized: using '{first_key}' as train split")
            return {'train': dataset[first_key]}
        
        return dataset
    
    def get_sample_count(self, dataset) -> int:
        """Get total sample count from dataset, handling various structures"""
        if not dataset:
            return 0
        
        if isinstance(dataset, dict):
            # Try common split names in order of preference
            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset:
                    return len(dataset[split_name])
            
            # If no common split names, use the first available
            if dataset:
                first_key = next(iter(dataset.keys()))
                return len(dataset[first_key])
        
        # If dataset is not a dict, assume it's a single Dataset object
        return len(dataset)

def main():
    """Demo script for dataset loader"""
    print("🚀 Drone Dataset Loader Demo")
    
    # Initialize loader
    loader = DroneDatasetLoader()
    
    # Load dataset
    dataset = loader.load_raw_dataset()
    
    # Show info
    loader.print_dataset_info(dataset)
    
    dataset_fraction = 1/25
    sampled_dataset = loader.sample_dataset(dataset, fraction=dataset_fraction)
    print(f"\n🔽 After sampling:")
    loader.print_dataset_info(sampled_dataset)
    
    # Create splits
    splits = loader.prepare_dataset_splits(sampled_dataset)
    
    # Save processed dataset
    loader.save_processed_dataset(splits, f"drone_sampled_{dataset_fraction:.2f}")
    
    print("✅ Dataset processing complete!")

if __name__ == "__main__":
    main()