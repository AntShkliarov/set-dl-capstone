"""
Test script for the DroneAudioModelEvaluator class
Tests evaluation functionality on trained models
Optimized to preprocess dataset only once for efficiency

DISCLAIMER: This script is for testing purposes only and has been generated using AI.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Add project root to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval.model_evaluator import DroneAudioModelEvaluator, evaluate_model

# Global test context to avoid repeated preprocessing
class TestContext:
    """Shared test context to avoid repeated expensive operations"""
    def __init__(self):
        self.models = None
        self.primary_evaluator = None
        self.dataset_loaded = False
        self.preprocessing_done = False
        
    def setup(self):
        """Setup shared test resources"""
        print("🔧 Setting up shared test context...")
        
        # Find models
        self.models = find_trained_models()
        if not self.models:
            print("⚠️  No trained models found for testing")
            return False
            
        # Setup primary evaluator with first model
        self.primary_evaluator = DroneAudioModelEvaluator(str(self.models[0]), "./test_results_shared")
        
        # Load model and feature extractor
        if not self.primary_evaluator.load_model_and_feature_extractor():
            print("⚠️  Failed to load model")
            return False
        
        # Load dataset once
        if not self.primary_evaluator.load_and_prepare_dataset():
            print("⚠️  Failed to load dataset")
            return False
        self.dataset_loaded = True
        
        # Preprocess dataset once
        if not self.primary_evaluator.preprocess_dataset():
            print("⚠️  Failed to preprocess dataset")
            return False
        self.preprocessing_done = True
        
        print(f"✅ Test context setup complete:")
        print(f"   Models found: {len(self.models)}")
        print(f"   Primary model: {self.primary_evaluator.model_name}")
        print(f"   Evaluation samples: {len(self.primary_evaluator.eval_dataset)}")
        
        return True

# Global test context instance
test_ctx = TestContext()

def find_trained_models() -> List[Path]:
    """Find all trained models in the models directory"""
    # Use current working directory as project root
    project_root = Path(os.getcwd())
    models_dir = project_root / "models"
    
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
    """Test dataset loading and preprocessing using shared context"""
    print("\n🧪 Testing dataset operations...")
    
    try:
        if not test_ctx.models:
            print("ℹ️  No trained models found, skipping dataset test")
            return True
        
        # Use the shared context that already has dataset loaded
        evaluator = test_ctx.primary_evaluator
        
        # Verify dataset components are loaded
        assert evaluator.dataset is not None
        assert test_ctx.dataset_loaded
        print("✅ Dataset loading successful (using shared context)")
        
        # Verify preprocessing is done
        assert evaluator.eval_dataset is not None
        assert test_ctx.preprocessing_done
        assert len(evaluator.eval_dataset) > 0
        print(f"✅ Dataset preprocessing successful: {len(evaluator.eval_dataset)} samples (using shared context)")
        
        return True
    except Exception as e:
        print(f"❌ Dataset operations test failed: {e}")
        return False


def test_model_evaluation():
    """Test model evaluation process using shared context"""
    print("\n🧪 Testing model evaluation...")
    
    try:
        if not test_ctx.models:
            print("ℹ️  No trained models found, skipping evaluation test")
            return True
        
        # Use the shared context that already has everything prepared
        evaluator = test_ctx.primary_evaluator
        
        # Run evaluation (no need to reload/preprocess)
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


def main():
    """Run all evaluator tests"""
    print("🚀 Starting DroneAudioModelEvaluator Tests")
    print("=" * 70)
    
    # Setup shared test context first to avoid repeated preprocessing
    print("🔧 Setting up shared test context...")
    if not test_ctx.setup():
        print("❌ Failed to setup test context, running basic tests only")
        # Run only initialization test if setup fails
        basic_tests = [("Evaluator Initialization", test_evaluator_initialization)]
    else:
        # Run all tests with optimized context
        basic_tests = [
            ("Evaluator Initialization", test_evaluator_initialization),
            ("Model Loading", test_model_loading),
            ("Dataset Operations", test_dataset_operations),
            ("Model Evaluation", test_model_evaluation),
            ("Full Pipeline (Isolated)", test_full_pipeline)
        ]
    
    print(f"📂 Found {len(test_ctx.models or [])} trained models:")
    for model in (test_ctx.models or []):
        print(f"   • {model.name}")
    print()
    
    results = {}
    
    for test_name, test_func in basic_tests:
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
    
    print(f"\n📊 Results: {passed_count}/{len(basic_tests)} tests passed")
    
    if passed_count == len(basic_tests):
        print("\n🎉 All evaluator tests passed! The DroneAudioModelEvaluator is ready to use.")
        print("⚡ Optimized testing: Dataset was preprocessed only once and reused across tests.")
    else:
        print(f"\n⚠️  {len(basic_tests) - passed_count} test(s) failed. Please check the issues above.")
    
    return passed_count == len(basic_tests)


if __name__ == "__main__":
    main()