#!/usr/bin/env python3
"""
Demo script for drone sound predictions using files from the sounds folder
Tests the prediction system with available models on real audio files
"""

from pathlib import Path
from predict.predict_drone_sounds import DroneAudioPredictor, list_available_models

def get_sounds_files():
    """Get all audio files from the sounds folder"""
    sounds_dir = Path('./sounds')
    if not sounds_dir.exists():
        print("❌ Sounds directory not found")
        return []
    
    audio_files = list(sounds_dir.glob('*.wav'))
    return sorted(audio_files)

def categorize_audio_files(audio_files):
    """Categorize audio files based on their names"""
    drone_files = []
    non_drone_files = []
    unknown_files = []
    
    for file in audio_files:
        filename = file.name.lower()
        
        # Files that likely contain drone sounds
        if any(keyword in filename for keyword in ['bebop', 'drone', 'd1', 'd2', 'membo']):
            drone_files.append(file)
        # Files that likely don't contain drone sounds (numbered files might be background/environment)
        elif filename.startswith('1-'):
            non_drone_files.append(file)
        else:
            unknown_files.append(file)
    
    return drone_files, non_drone_files, unknown_files

def demo_with_sounds_folder():
    """Run demo with audio files from the sounds folder"""
    print("🚁 Drone Sound Prediction Demo")
    print("🎵 Using audio files from ./sounds folder")
    print("=" * 60)
    
    # Get available models
    models = list_available_models()
    if not models:
        print("❌ No trained models found")
        return
    
    # Use the best available model (AST if available, as it has highest accuracy)
    ast_model = None
    wav2vec2_model = None
    hubert_model = None
    
    for model in models:
        if "ast" in model.name.lower():
            ast_model = model
        elif "wav2vec2" in model.name.lower():
            wav2vec2_model = model
        elif "hubert" in model.name.lower():
            hubert_model = model
    
    # Choose best model - prioritize AST (100% accuracy) for this demo
    if ast_model:
        model_path = ast_model
        print(f"🎯 Using AST model (100.0% accuracy)")
    elif wav2vec2_model:
        model_path = wav2vec2_model
        print(f"🎯 Using Wav2Vec2 model (99.91% accuracy)")
    elif hubert_model:
        model_path = hubert_model
        print(f"🎯 Using HuBERT model (99.82% accuracy)")
    else:
        model_path = models[0]
        print(f"🎯 Using available model: {model_path.name}")
    
    # Get audio files from sounds folder
    audio_files = get_sounds_files()
    if not audio_files:
        print("❌ No audio files found in ./sounds folder")
        return
    
    print(f"\n📂 Found {len(audio_files)} audio files in ./sounds folder:")
    for i, file in enumerate(audio_files):
        print(f"   {i+1:2d}. {file.name}")
    
    # Categorize files
    drone_files, non_drone_files, unknown_files = categorize_audio_files(audio_files)
    
    print(f"\n📊 File categorization:")
    print(f"   🚁 Likely drone sounds: {len(drone_files)} files")
    print(f"   🌍 Likely background/env: {len(non_drone_files)} files")
    print(f"   ❓ Unknown category: {len(unknown_files)} files")
    
    try:
        # Initialize predictor
        predictor = DroneAudioPredictor(str(model_path))
        
        # Test all files
        results_summary = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'drone_detected': 0,
            'detailed_results': []
        }
        
        for category_name, files, expected_drone in [
            ("🚁 DRONE SOUNDS", drone_files, True),
            ("🌍 BACKGROUND/ENVIRONMENT", non_drone_files, False),
            ("❓ UNKNOWN CATEGORY", unknown_files, None)
        ]:
            if not files:
                continue
                
            print(f"\n" + "─" * 60)
            print(f"{category_name} ({len(files)} files)")
            print("─" * 60)
            
            for audio_file in files:
                try:
                    print(f"\n🎵 Testing: {audio_file.name}")
                    results = predictor.predict_file(str(audio_file))
                    
                    # Display results
                    drone_detected = results['is_drone']
                    confidence = results['confidence']
                    
                    print(f"   🎯 Prediction: {results['predicted_label']}")
                    print(f"   📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                    print(f"   🚁 Drone Detected: {'YES' if drone_detected else 'NO'}")
                    
                    # Check accuracy if we have expected result
                    if expected_drone is not None:
                        correct = expected_drone == drone_detected
                        status = "✅ CORRECT" if correct else "❌ INCORRECT"
                        print(f"   {status} (Expected: {'Drone' if expected_drone else 'Non-Drone'})")
                        
                        if correct:
                            results_summary['correct_predictions'] += 1
                        results_summary['total_predictions'] += 1
                    
                    if drone_detected:
                        results_summary['drone_detected'] += 1
                    
                    # Store detailed results
                    results_summary['detailed_results'].append({
                        'file': audio_file.name,
                        'category': category_name,
                        'predicted_drone': drone_detected,
                        'confidence': confidence,
                        'expected_drone': expected_drone,
                        'correct': expected_drone == drone_detected if expected_drone is not None else None
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing {audio_file.name}: {e}")
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        print(f"📊 Total files tested: {len(audio_files)}")
        print(f"🚁 Drone sounds detected: {results_summary['drone_detected']}")
        print(f"🌍 Non-drone sounds: {len(audio_files) - results_summary['drone_detected']}")
        
        if results_summary['total_predictions'] > 0:
            accuracy = results_summary['correct_predictions'] / results_summary['total_predictions']
            print(f"🎯 Accuracy on categorized files: {accuracy:.1%} ({results_summary['correct_predictions']}/{results_summary['total_predictions']})")
        
        print(f"\n🤖 Model used: {predictor.model_name}")
        print("=" * 60)
        
        # Show breakdown by file type
        print(f"\n📋 Detailed breakdown:")
        for result in results_summary['detailed_results']:
            status = ""
            if result['correct'] is not None:
                status = " ✅" if result['correct'] else " ❌"
            print(f"   {result['file']:<30} → {'Drone' if result['predicted_drone'] else 'Non-Drone'} ({result['confidence']:.2f}){status}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_with_sounds_folder()