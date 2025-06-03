#!/usr/bin/env python3
"""
Model evaluation script using the DroneAudioModelEvaluator class
Demonstrates evaluation of trained models with the enhanced evaluator functionality
"""

from pathlib import Path
from src.eval.model_evaluator import (
    DroneAudioModelEvaluator,
    evaluate_model,
    find_all_trained_models,
    find_latest_checkpoint,
    evaluate_all_models,
    compare_model_performance
)


def evaluate_wav2vec2_model():
    """Evaluate the Wav2Vec2 model using the evaluator class"""
    print("🎯 Evaluating Wav2Vec2 Model")
    print("=" * 50)
    
    model_path = "./models/facebook_wav2vec2-base_drone_classifier"
    
    if not Path(model_path).exists():
        print(f"❌ Wav2Vec2 model not found at {model_path}")
        return None
    
    # Using DroneAudioModelEvaluator class directly
    print("📊 Using DroneAudioModelEvaluator class")
    evaluator = DroneAudioModelEvaluator(model_path, "./results")
    results, viz_files = evaluator.run_full_evaluation()
    
    return results


def evaluate_hubert_model():
    """Evaluate the HuBERT model if available"""
    print("\n🎯 Evaluating HuBERT Model")
    print("=" * 50)
    
    model_path = "./models/facebook_hubert-base-ls960_drone_classifier"
    
    if not Path(model_path).exists():
        print(f"❌ HuBERT model not found at {model_path}")
        return None
    
    # Using the convenience function
    print("📊 Using convenience function")
    results, viz_files = evaluate_model(model_path, results_dir="./results")
    
    return results


def evaluate_ast_model():
    """Evaluate the AST model using latest checkpoint detection"""
    print("\n🎯 Evaluating AST (Audio Spectrogram Transformer) Model")
    print("=" * 50)
    
    base_model_path = Path("./models/MIT_ast-finetuned-audioset-10-10-0.4593_drone_classifier")
    
    if not base_model_path.exists():
        print(f"❌ AST model not found at {base_model_path}")
        return None
    
    # Use enhanced model evaluator with automatic checkpoint detection
    model_path = find_latest_checkpoint(base_model_path)
    
    print("📊 Using DroneAudioModelEvaluator with automatic checkpoint detection")
    evaluator = DroneAudioModelEvaluator(str(model_path), "./results")
    results, viz_files = evaluator.run_full_evaluation()
    
    return results


def main():
    """Main evaluation script - demonstrates enhanced DroneAudioModelEvaluator functionality"""
    print("🚀 Drone Audio Model Evaluation Suite")
    print("=" * 60)
    print("Using enhanced DroneAudioModelEvaluator functionality")
    print("=" * 60)
    
    # Evaluate specific models using enhanced functionality
    print("\n🎯 Model Evaluation")
    results_list = []
    
    wav2vec2_results = evaluate_wav2vec2_model()
    if wav2vec2_results:
        results_list.append({
            'model_name': wav2vec2_results['model_name'],
            'results': wav2vec2_results
        })
    
    hubert_results = evaluate_hubert_model()
    if hubert_results:
        results_list.append({
            'model_name': hubert_results['model_name'],
            'results': hubert_results
        })
    
    ast_results = evaluate_ast_model()
    if ast_results:
        results_list.append({
            'model_name': ast_results['model_name'],
            'results': ast_results
        })
    
    # Compare model performance using enhanced functionality
    if results_list:
        compare_model_performance(results_list)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("📂 Results and visualizations saved to: ./results/")
    print("📊 Enhanced DroneAudioModelEvaluator provides:")
    print("   • Automatic checkpoint detection")
    print("   • Performance comparison utilities")
    print("   • Comprehensive visualization generation")
    print("=" * 60)


if __name__ == "__main__":
    main()