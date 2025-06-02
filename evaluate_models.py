#!/usr/bin/env python3
"""
Model evaluation script using the DroneAudioModelEvaluator class
Demonstrates evaluation of trained models with the new reusable evaluator
"""

from pathlib import Path
from src.eval.model_evaluator import DroneAudioModelEvaluator, evaluate_model


def find_all_trained_models():
    """Find all trained models in the models directory"""
    models_dir = Path("./models")
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return []
    
    model_dirs = []
    for item in models_dir.iterdir():
        if item.is_dir() and "drone_classifier" in item.name:
            # Check if it has required files
            if (item / "config.json").exists() and (item / "model.safetensors").exists():
                model_dirs.append(item)
    
    return sorted(model_dirs)


def evaluate_wav2vec2_model():
    """Evaluate the Wav2Vec2 model using the new evaluator class"""
    print("🎯 Evaluating Wav2Vec2 Model")
    print("=" * 50)
    
    model_path = "./models/facebook_wav2vec2-base_drone_classifier"
    
    if not Path(model_path).exists():
        print(f"❌ Wav2Vec2 model not found at {model_path}")
        return None
    
    # Method 1: Using the class directly
    print("📊 Method 1: Using DroneAudioModelEvaluator class")
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
    
    # Method 2: Using the convenience function
    print("📊 Method 2: Using convenience function")
    results, viz_files = evaluate_model(model_path, results_dir="./results")
    
    return results


def evaluate_ast_model():
    """Evaluate the AST model at its latest checkpoint using the new evaluator class"""
    print("\n🎯 Evaluating AST (Audio Spectrogram Transformer) Model")
    print("=" * 50)
    
    base_model_path = "./models/MIT_ast-finetuned-audioset-10-10-0.4593_drone_classifier"
    
    if not Path(base_model_path).exists():
        print(f"❌ AST model not found at {base_model_path}")
        return None
    
    # Find the latest checkpoint
    checkpoints = list(Path(base_model_path).glob("checkpoint-*"))
    if checkpoints:
        # Use the latest checkpoint (highest number)
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split('-')[1]))[-1]
        model_path = str(latest_checkpoint)
        print(f"📌 Using latest checkpoint: {latest_checkpoint.name}")
    else:
        # Fall back to base model if no checkpoints found
        model_path = base_model_path
        print("📌 Using base model (no checkpoints found)")
    
    # Method 3: Using the class directly for AST model
    print("📊 Method 3: Using DroneAudioModelEvaluator class for AST")
    evaluator = DroneAudioModelEvaluator(model_path, "./results")
    results, viz_files = evaluator.run_full_evaluation()
    
    return results


def evaluate_all_available_models():
    """Evaluate all available trained models"""
    print("\n🎯 Evaluating All Available Models")
    print("=" * 50)
    
    models = find_all_trained_models()
    
    if not models:
        print("❌ No trained models found")
        return []
    
    print(f"📂 Found {len(models)} trained models:")
    for model in models:
        print(f"   • {model.name}")
    
    all_results = []
    
    for i, model_path in enumerate(models):
        print(f"\n📊 Evaluating {i+1}/{len(models)}: {model_path.name}")
        print("-" * 40)
        
        try:
            # Use convenience function for batch evaluation
            results, viz_files = evaluate_model(str(model_path), results_dir="./results")
            all_results.append({
                'model_path': str(model_path),
                'model_name': results['model_name'],
                'results': results,
                'viz_files': viz_files
            })
            
            print(f"✅ {results['model_name']} evaluated successfully")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1 Score: {results['f1_score']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall: {results['recall']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_path.name}: {e}")
    
    return all_results


def compare_model_performance(all_results):
    """Compare performance across all evaluated models"""
    print("\n🏆 Model Performance Comparison")
    print("=" * 50)
    
    if not all_results:
        print("❌ No results to compare")
        return
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x['results']['accuracy'], reverse=True)
    
    print("📊 Rankings by Accuracy:")
    for i, result in enumerate(sorted_results):
        results = result['results']
        print(f"{i+1:2d}. {results['model_name']:<35} "
              f"Acc: {results['accuracy']:.4f} "
              f"F1: {results['f1_score']:.4f} "
              f"Prec: {results['precision']:.4f} "
              f"Rec: {results['recall']:.4f}")
    
    # Highlight best performing model
    best_model = sorted_results[0]
    best_results = best_model['results']
    
    print(f"\n🥇 Best Performing Model: {best_results['model_name']}")
    print(f"   📈 Accuracy: {best_results['accuracy']:.4f}")
    print(f"   📈 F1 Score: {best_results['f1_score']:.4f}")
    print(f"   📈 Precision: {best_results['precision']:.4f}")
    print(f"   📈 Recall: {best_results['recall']:.4f}")
    
    if 'roc_data' in best_results and best_results['roc_data']['auc'] is not None:
        print(f"   📈 AUC Score: {best_results['roc_data']['auc']:.4f}")


def main():
    """Main evaluation script"""
    print("🚀 Drone Audio Model Evaluation Suite")
    print("=" * 60)
    print("Using DroneAudioModelEvaluator for comprehensive model evaluation")
    print("=" * 60)
    
    # Option 1: Evaluate specific models
    print("\n🎯 Option 1: Specific Model Evaluation")
    wav2vec2_results = evaluate_wav2vec2_model()
    hubert_results = evaluate_hubert_model()
    ast_results = evaluate_ast_model()
    all_results = [wav2vec2_results, hubert_results, ast_results]
    
    # Option 3: Compare model performance
    if all_results:
        compare_model_performance(all_results)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("📂 Results and visualizations saved to: ./results/")
    print("📊 Use the DroneAudioModelEvaluator class for custom evaluations")
    print("=" * 60)


if __name__ == "__main__":
    main()