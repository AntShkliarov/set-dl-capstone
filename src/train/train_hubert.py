#!/usr/bin/env python3
"""
HuBERT Training Script for Drone Sound Recognition
Using Facebook's HuBERT model from Hugging Face for audio classification
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from drone_pipeline import DroneAudioPipeline
from transformers import TrainingArguments

def main():
    """Train HuBERT model for drone sound recognition."""
    
    print("🎵 Starting HuBERT Training for Drone Sound Recognition")
    print("=" * 80)
    
    # Model configuration
    model_name = "facebook/hubert-base-ls960"
    dataset_name = "drone_sampled_0.04"
    
    print(f"📱 Model: {model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Hardware: Apple M4 Pro with MPS acceleration")
    print()
    
    # Custom training arguments for HuBERT with reduced epochs
    custom_training_args = {
        "num_train_epochs": 5,  # Reduced from default 10 for faster training
        "per_device_train_batch_size": 8,  # Optimal for HuBERT audio features
        "per_device_eval_batch_size": 8,
        "learning_rate": 3e-5,  # Standard learning rate for HuBERT fine-tuning
        "warmup_steps": 100,
        "logging_steps": 50,
        "eval_steps": 500,
        "save_steps": 500,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "report_to": None,  # Disable wandb logging
        "push_to_hub": False,
        "dataloader_num_workers": 2,
        "weight_decay": 0.01,
    }
    
    print("🎯 Training Configuration:")
    print(f"   • Epochs: {custom_training_args['num_train_epochs']}")
    print(f"   • Batch Size: {custom_training_args['per_device_train_batch_size']}")
    print(f"   • Learning Rate: {custom_training_args['learning_rate']}")
    print(f"   • Evaluation Strategy: {custom_training_args['evaluation_strategy']}")
    print()
    
    try:
        # Initialize pipeline
        print("🔄 Initializing DroneAudioPipeline...")
        pipeline = DroneAudioPipeline(
            model_name=model_name
        )
        
        model, trainer, results = pipeline.run_full_pipeline(
            dataset_name=dataset_name,
            custom_training_args=custom_training_args
        )
        
        print("✅ HuBERT Training Pipeline Completed Successfully!")
        print()
        print("📈 Final Results:")
        print(f"   • Accuracy: {results['accuracy']:.4f}")
        print(f"   • Precision: {results['precision']:.4f}")
        print(f"   • Recall: {results['recall']:.4f}")
        print(f"   • F1 Score: {results['f1_score']:.4f}")
        if results['roc_data']['auc']:
            print(f"   • AUC Score: {results['roc_data']['auc']:.4f}")
        print()
        print("📁 Results saved to:")
        print(f"   • Model: models/{model_name.replace('/', '_')}_drone_classifier/")
        print(f"   • Metrics: results/{model_name.replace('/', '_')}_results.pkl")
        print(f"   • Visualizations: results/{model_name.replace('/', '_')}_*.png")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during HuBERT training: {str(e)}")
        raise

if __name__ == "__main__":
    main()