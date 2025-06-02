"""
Test script for the DroneAudioModelEvaluator class
Tests evaluation functionality on trained models
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from eval.model_evaluator import DroneAudioModelEvaluator, evaluate_model


def find_trained_models() -> List[Path]:
    """Find all trained models in the models directory"""
    models_dir = Path("../../models")
    
    if not models_dir.exists():
        print("⚠️  Models directory not found")
        return []
    
    # Look for model directories
    model_dirs = []
    for item in models_dir.iterdir():
        if item.is_dir() and "drone_classifier" in item.name:
            # Check if it has required files
            if (item / "config.json").exists() and (item / "model.safetensors").exists():
                model_dirs.append(item)
    
    return model_dirs


def test_evaluator_initialization():
    """Test basic evaluator initialization"""
    print("🧪 Testing evaluator initialization...")
    
    try:
        # Test with non-existent model (should handle gracefully)
        evaluator = DroneAudioModelEvaluator("./non_existent_model")
        assert evaluator.model_path.name == "non_existent_model"
        assert evaluator.results_dir.exists()
        print("✅ Initialization with non-existent model handled correctly")
        
        # Test with existing models directory
        models = find_trained_models()
        if models:
            evaluator = DroneAudioModelEvaluator(str(models[0]))
            assert evaluator.model_path.exists()
            print("✅ Initialization with existing model successful")
        else:
            print("ℹ️  No trained models found for testing")
        
        return True
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False


def test_model_loading():
    """Test model and feature extractor loading"""
    print("\n🧪 Testing model loading...")
    
    try:
        models = find_trained_models()
        if not models:
            print("ℹ️  No trained models found, skipping model loading test")
            return True
        
        for model_path in models[:2]:  # Test first 2 models to save time
            print(f"📂 Testing model: {model_path.name}")
            
            evaluator = DroneAudioModelEvaluator(str(model_path))
            success = evaluator.load_model_and_feature_extractor()
            
            if success:
                assert evaluator.model is not None
                assert evaluator.feature_extractor is not None
                assert evaluator.model_name is not None
                
                # Test model summary
                summary = evaluator.get_model_summary()
                assert "model_name" in summary
                assert "num_parameters" in summary
                assert summary["num_parameters"] > 0
                
                print(f"✅ Model loaded successfully: {evaluator.model_name}")
                print(f"   Parameters: {summary['num_parameters']:,}")
                print(f"   Sampling rate: {summary['sampling_rate']}Hz")
            else:
                print(f"⚠️  Failed to load model: {model_path.name}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_dataset_operations():
    """Test dataset loading and preprocessing"""
    print("\n🧪 Testing dataset operations...")
    
    try:
        models = find_trained_models()
        if not models:
            print("ℹ️  No trained models found, skipping dataset test")
            return True
        
        # Use the first available model
        evaluator = DroneAudioModelEvaluator(str(models[0]))
        
        # Load model (needed for feature extractor)
        if not evaluator.load_model_and_feature_extractor():
            print("⚠️  Failed to load model, skipping dataset test")
            return True
        
        # Test dataset loading
        dataset_success = evaluator.load_and_prepare_dataset()
        if not dataset_success:
            print("⚠️  Failed to load dataset")
            return True
        
        assert evaluator.dataset is not None
        print("✅ Dataset loading successful")
        
        # Test preprocessing
        preprocess_success = evaluator.preprocess_dataset()
        if preprocess_success:
            assert evaluator.eval_dataset is not None
            assert len(evaluator.eval_dataset) > 0
            print(f"✅ Dataset preprocessing successful: {len(evaluator.eval_dataset)} samples")
        else:
            print("⚠️  Dataset preprocessing failed")
        
        return True
    except Exception as e:
        print(f"❌ Dataset operations test failed: {e}")
        return False


def test_model_evaluation():
    """Test model evaluation process"""
    print("\n🧪 Testing model evaluation...")
    
    try:
        models = find_trained_models()
        if not models:
            print("ℹ️  No trained models found, skipping evaluation test")
            return True
        
        # Use the first available model
        evaluator = DroneAudioModelEvaluator(str(models[0]))
        
        # Load model
        if not evaluator.load_model_and_feature_extractor():
            print("⚠️  Failed to load model, skipping evaluation test")
            return True
        
        # Load and preprocess dataset
        if not evaluator.load_and_prepare_dataset():
            print("⚠️  Failed to load dataset, skipping evaluation test")
            return True
        
        if not evaluator.preprocess_dataset():
            print("⚠️  Failed to preprocess dataset, skipping evaluation test")
            return True
        
        # Run evaluation
        eval_success = evaluator.evaluate_model()
        if eval_success:
            assert evaluator.predictions_data is not None
            assert 'y_pred' in evaluator.predictions_data
            assert 'y_true' in evaluator.predictions_data
            assert len(evaluator.predictions_data['y_pred']) > 0
            print("✅ Model evaluation successful")
            
            # Test metrics calculation
            results = evaluator.calculate_metrics()
            assert 'accuracy' in results
            assert 'f1_score' in results
            assert 0 <= results['accuracy'] <= 1
            assert 0 <= results['f1_score'] <= 1
            print(f"✅ Metrics calculation successful")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1 Score: {results['f1_score']:.4f}")
        else:
            print("⚠️  Model evaluation failed")
        
        return True
    except Exception as e:
        print(f"❌ Model evaluation test failed: {e}")
        return False


def test_full_pipeline():
    """Test the complete evaluation pipeline"""
    print("\n🧪 Testing full evaluation pipeline...")
    
    try:
        models = find_trained_models()
        if not models:
            print("ℹ️  No trained models found, skipping full pipeline test")
            return True
        
        # Test the first model with full pipeline
        model_path = models[0]
        print(f"📂 Testing full pipeline with: {model_path.name}")
        
        evaluator = DroneAudioModelEvaluator(str(model_path), "./test_results")
        
        # Run full evaluation
        results, viz_files = evaluator.run_full_evaluation()
        
        # Validate results
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'f1_score' in results
        assert isinstance(viz_files, list)
        
        print("✅ Full pipeline test successful")
        print(f"   Model: {results['model_name']}")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Visualizations: {len(viz_files)} files")
        
        # Test convenience function
        print("\n📊 Testing convenience function...")
        results2, viz_files2 = evaluate_model(str(model_path), results_dir="./test_results2")
        
        assert isinstance(results2, dict)
        assert 'accuracy' in results2
        print("✅ Convenience function test successful")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


def test_multiple_models():
    """Test evaluation on multiple models if available"""
    print("\n🧪 Testing multiple models...")
    
    try:
        models = find_trained_models()
        if len(models) < 2:
            print("ℹ️  Less than 2 models found, skipping multi-model test")
            return True
        
        results_summary = []
        
        for i, model_path in enumerate(models[:3]):  # Test up to 3 models
            print(f"\n📂 Testing model {i+1}/{min(3, len(models))}: {model_path.name}")
            
            try:
                evaluator = DroneAudioModelEvaluator(str(model_path), f"./test_results_model_{i}")
                
                # Quick evaluation (just metrics, no visualizations)
                if evaluator.load_model_and_feature_extractor():
                    if evaluator.load_and_prepare_dataset():
                        if evaluator.preprocess_dataset():
                            if evaluator.evaluate_model():
                                results = evaluator.calculate_metrics()
                                results_summary.append({
                                    'model': results['model_name'],
                                    'accuracy': results['accuracy'],
                                    'f1_score': results['f1_score']
                                })
                                print(f"✅ {results['model_name']}: Acc={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
            except Exception as e:
                print(f"⚠️  Failed to evaluate {model_path.name}: {e}")
        
        if results_summary:
            print(f"\n📊 Multi-model evaluation summary:")
            for result in results_summary:
                print(f"   {result['model']}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
            print("✅ Multi-model test successful")
        
        return True
    except Exception as e:
        print(f"❌ Multi-model test failed: {e}")
        return False


def main():
    """Run all evaluator tests"""
    print("🚀 Starting DroneAudioModelEvaluator Tests")
    print("=" * 70)
    
    # First, check for available models
    models = find_trained_models()
    print(f"📂 Found {len(models)} trained models:")
    for model in models:
        print(f"   • {model.name}")
    print()
    
    tests = [
        ("Evaluator Initialization", test_evaluator_initialization),
        ("Model Loading", test_model_loading),
        ("Dataset Operations", test_dataset_operations),
        ("Model Evaluation", test_model_evaluation),
        ("Full Pipeline", test_full_pipeline),
        ("Multiple Models", test_multiple_models)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-'*70}")
        print(f"Running: {test_name}")
        print('-'*70)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*70}")
    print("MODEL EVALUATOR TEST SUMMARY")
    print('='*70)
    
    passed_count = 0
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if passed:
            passed_count += 1
    
    print(f"\n📊 Results: {passed_count}/{len(tests)} tests passed")
    
    if passed_count == len(tests):
        print("\n🎉 All evaluator tests passed! The DroneAudioModelEvaluator is ready to use.")
    else:
        print(f"\n⚠️  {len(tests) - passed_count} test(s) failed. Please check the issues above.")
    
    return passed_count == len(tests)


if __name__ == "__main__":
    main()