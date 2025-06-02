#!/usr/bin/env python3
"""
Wav2Vec2 Training Script for Drone Sound Recognition
Using Facebook's Wav2Vec2 model from Hugging Face for audio classification
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from drone_pipeline import DroneAudioPipeline
from transformers import TrainingArguments

def main():
    """Train Wav2Vec2 model for drone sound recognition."""
    
    print("🎤 Starting Wav2Vec2 Training for Drone Sound Recognition")
    print("=" * 80)
    
    # Model configuration
    model_name = "facebook/wav2vec2-base"
    dataset_name = "drone_sampled_0.04"
    
    print(f"📱 Model: {model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Hardware: Apple M4 Pro with MPS acceleration")
    print()
    
    # Custom training arguments for Wav2Vec2 with standard epochs
    custom_training_args = {
        "num_train_epochs": 10,  # Standard 10 epochs for comprehensive training
        "per_device_train_batch_size": 8,  # Optimal for Wav2Vec2 audio features
        "per_device_eval_batch_size": 8,
        "learning_rate": 3e-5,  # Standard learning rate for Wav2Vec2 fine-tuning
        "warmup_steps": 100,
        "logging_steps": 10,  # More frequent logging for detailed monitoring
        "eval_steps": 500,
        "save_steps": 500,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
        "report_to": None,  # Disable wandb logging
        "push_to_hub": False,
        "dataloader_num_workers": 2,
        "weight_decay": 0.01,
        "save_total_limit": 2,
    }
    
    print("🎯 Training Configuration:")
    print(f"   • Epochs: {custom_training_args['num_train_epochs']}")
    print(f"   • Batch Size: {custom_training_args['per_device_train_batch_size']}")
    print(f"   • Learning Rate: {custom_training_args['learning_rate']}")
    print(f"   • Evaluation Strategy: {custom_training_args['evaluation_strategy']}")
    print(f"   • Best Model Metric: {custom_training_args['metric_for_best_model']}")
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
        
        print("✅ Wav2Vec2 Training Pipeline Completed Successfully!")
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
        print(f"❌ Error during Wav2Vec2 training: {str(e)}")
        raise

if __name__ == "__main__":
    main()