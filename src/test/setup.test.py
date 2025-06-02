"""
Test setup script to verify dataset loading and basic functionality
"""

from audio_classification import DroneAudioClassifier
import torch

def test_dataset_loading():
    """Test basic dataset loading"""
    print("🧪 Testing dataset loading...")
    
    classifier = DroneAudioClassifier()
    
    try:
        # Test dataset loading
        dataset = classifier.load_drone_dataset()
        print(f"✅ Dataset loaded successfully: {len(dataset['train'])} samples")
        
        # Inspect first sample
        sample = dataset['train'][0]
        print(f"✅ Sample structure: {list(sample.keys())}")
        
        # Test audio data
        audio_data = sample['audio']
        print(f"✅ Audio info:")
        print(f"   - Sampling rate: {audio_data['sampling_rate']}")
        print(f"   - Array shape: {len(audio_data['array'])}")
        print(f"   - Duration: {len(audio_data['array']) / audio_data['sampling_rate']:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\n🧪 Testing preprocessing...")
    
    classifier = DroneAudioClassifier()
    
    try:
        # Load dataset
        dataset = classifier.load_drone_dataset()
        
        # Test preprocessing with small sample
        small_dataset = dataset['train'].select(range(min(10, len(dataset['train']))))
        processed_dataset, feature_extractor = classifier.prepare_dataset_for_model(small_dataset)
        
        print(f"✅ Preprocessing successful")
        print(f"   - Original features: {dataset['train'].features}")
        print(f"   - Processed features: {processed_dataset.features}")
        print(f"   - Feature extractor: {feature_extractor}")
        
        # Test a single batch
        batch = processed_dataset[:2]
        print(f"   - Batch keys: {list(batch.keys())}")
        if 'input_values' in batch:
            print(f"   - Input values type: {type(batch['input_values'])}")
            print(f"   - Input values length: {len(batch['input_values'])}")
            if len(batch['input_values']) > 0:
                print(f"   - First input shape: {len(batch['input_values'][0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_device_setup():
    """Test device setup for Apple Silicon"""
    print("\n🧪 Testing device setup...")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU) is available")
        device = torch.device("mps")
        
        # Test basic tensor operations on MPS
        try:
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = torch.mm(x, y)
            print("✅ Basic MPS operations working")
        except Exception as e:
            print(f"⚠️  MPS available but operations failed: {e}")
            print("Using CPU instead")
    else:
        print("⚠️  MPS not available, using CPU")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting setup verification tests...\n")
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Preprocessing", test_preprocessing),
        ("Device Setup", test_device_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Ready to start training.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()