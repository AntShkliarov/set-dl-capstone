#!/usr/bin/env python3
"""
Audio Spectrogram Transformer (AST) Training Script for Drone Sound Recognition
Using MIT's AST model from Hugging Face for audio classification
"""

import os
import sys
from pathlib import Path

# Add project root to Python path (go up two levels from src/train/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.drone_pipeline import DroneAudioPipeline
from transformers import TrainingArguments

def main():
    """Train AST model for drone sound recognition."""
    
    print("🚁 Starting Audio Spectrogram Transformer (AST) Training for Drone Sound Recognition")
    print("=" * 80)
    
    # Model configuration
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    dataset_name = "drone_sampled_0.04"
    
    print(f"📱 Model: {model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Hardware: Apple M4 Pro with MPS acceleration")
    print()
    
    # Custom training arguments for AST with reduced epochs
    custom_training_args = {
        "num_train_epochs": 5,  # Reduced from default 10 for faster training
        "per_device_train_batch_size": 8,  # Smaller batch size for AST spectrograms
        "per_device_eval_batch_size": 8,
        "learning_rate": 3e-5,  # Slightly higher learning rate for AST
        "warmup_steps": 100,
        "logging_steps": 50,
        "eval_strategy": "epoch",  # Evaluate at end of each epoch (more consistent)
        "save_strategy": "epoch",  # Save at end of each epoch
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "report_to": None,  # Disable wandb logging
        "push_to_hub": False,
        "dataloader_num_workers": 2,
    }
    
    print("🎯 Training Configuration:")
    print(f"   • Epochs: {custom_training_args['num_train_epochs']}")
    print(f"   • Batch Size: {custom_training_args['per_device_train_batch_size']}")
    print(f"   • Learning Rate: {custom_training_args['learning_rate']}")
    print(f"   • Evaluation Strategy: {custom_training_args['eval_strategy']}")
    print()
    
    try:
        # Initialize pipeline
        print("🔄 Initializing DroneAudioPipeline...")
        pipeline = DroneAudioPipeline(model_name=model_name)
        
        # Run full pipeline with dataset and custom training args
        pipeline.run_full_pipeline(
            dataset_name=dataset_name,
            custom_training_args=custom_training_args,
            use_advanced_evaluator=True
        )
        
        print("✅ AST Training Pipeline Completed Successfully!")
        print()
        print("📈 Results saved to:")
        print(f"   • Model: models/{model_name.replace('/', '_')}_drone_classifier/")
        print(f"   • Metrics: results/{model_name.replace('/', '_')}_results.pkl")
        print(f"   • Visualizations: results/{model_name.replace('/', '_')}_*.png")
        
    except Exception as e:
        print(f"❌ Error during AST training: {str(e)}")
        raise

if __name__ == "__main__":
    main()