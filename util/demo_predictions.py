#!/usr/bin/env python3
"""
Demo script for drone sound predictions using files from the sounds folder
Tests the prediction system with available models on real audio files
"""

from predict.predict_drone_sounds import DroneAudioPredictor, get_best_available_model, find_audio_files

def get_sounds_files():
    """Get all audio files from the sounds folder"""
    audio_files = find_audio_files("./sounds")
    if not audio_files:
        print("❌ Sounds directory not found or no audio files")
        return []
    
    return audio_files


def demo_with_sounds_folder():
    """Run demo with audio files from the sounds folder"""
    print("🚁 Drone Sound Prediction Demo")
    print("=" * 50)
    
    # Get best model and audio files
    model_path = get_best_available_model()
    if not model_path:
        print("❌ No trained models found")
        return
    
    audio_files = get_sounds_files()
    if not audio_files:
        print("❌ No audio files found in ./sounds folder")
        return
    
    print(f"🎯 Using model: {model_path.name}")
    print(f"📂 Testing {len(audio_files)} audio files")
    
    try:
        predictor = DroneAudioPredictor(str(model_path))
        drone_count = 0
        
        print("\n🎵 Processing files...")
        for audio_file in audio_files:
            try:
                results = predictor.predict_file(str(audio_file))
                drone_detected = results['is_drone']
                confidence = results['confidence']
                
                status = "🚁 DRONE" if drone_detected else "🌍 NON-DRONE"
                print(f"   {audio_file.name:<25} → {status} ({confidence:.2f})")
                
                if drone_detected:
                    drone_count += 1
                    
            except Exception as e:
                print(f"   ⚠️  Error: {audio_file.name} - {e}")
        
        # Simple summary
        print(f"\n" + "=" * 50)
        print(f"📊 Results: {drone_count} drone sounds, {len(audio_files) - drone_count} non-drone")
        print(f"🤖 Model: {predictor.model_name}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_with_sounds_folder()