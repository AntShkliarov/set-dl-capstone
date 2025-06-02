#!/usr/bin/env python3
"""
Drone Sound Prediction Script
Uses librosa to load audio files and DroneAudioModelEvaluator for predictions
"""

import librosa
import numpy as np
import torch
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from src.eval.model_evaluator import DroneAudioModelEvaluator
import argparse

class DroneAudioPredictor:
    """Real-time drone audio prediction using trained models"""
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to trained model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_extractor = None
        self.model_name = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and feature extractor"""
        try:
            print(f"🔄 Loading model from {self.model_path}")
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_path)
            self.model_name = self.model_path.name.replace("_drone_classifier", "")
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"✅ Model loaded: {self.model_name}")
            print(f"   Expected sampling rate: {self.feature_extractor.sampling_rate}Hz")
            print(f"   Model classes: {self.model.config.id2label}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_audio_file(self, audio_path: str, target_duration: float = 10.0) -> np.ndarray:
        """
        Load and preprocess audio file using librosa
        
        Args:
            audio_path: Path to audio file
            target_duration: Target duration in seconds (default 10s)
            
        Returns:
            np.ndarray: Preprocessed audio array
        """
        try:
            print(f"🎵 Loading audio file: {audio_path}")
            
            # Load audio with librosa
            audio, sr = librosa.load(
                audio_path, 
                sr=self.feature_extractor.sampling_rate,  # Resample to model's expected rate
                duration=target_duration
            )
            
            # Ensure we have exactly the target duration
            target_length = int(self.feature_extractor.sampling_rate * target_duration)
            
            if len(audio) < target_length:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                # Truncate if too long
                audio = audio[:target_length]
            
            print(f"✅ Audio loaded and preprocessed:")
            print(f"   Original sampling rate: {sr}Hz")
            print(f"   Target sampling rate: {self.feature_extractor.sampling_rate}Hz")
            print(f"   Duration: {len(audio) / self.feature_extractor.sampling_rate:.2f}s")
            print(f"   Shape: {audio.shape}")
            
            return audio
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            raise
    
    def predict(self, audio: np.ndarray) -> dict:
        """
        Make prediction on audio array
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            dict: Prediction results with probabilities and confidence
        """
        try:
            print("🔮 Making prediction...")
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction results
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Map class index to label
            class_label = self.model.config.id2label[predicted_class]
            
            # Get all class probabilities
            all_probs = {
                self.model.config.id2label[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
            
            results = {
                'predicted_class': predicted_class,
                'predicted_label': class_label,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'is_drone': class_label.lower() in ['drone', '1', 'yes'] or predicted_class == 1
            }
            
            print(f"✅ Prediction complete:")
            print(f"   Predicted: {class_label}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Is Drone: {results['is_drone']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise
    
    def predict_file(self, audio_path: str) -> dict:
        """
        Complete pipeline: load audio file and make prediction
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Prediction results
        """
        audio = self.load_audio_file(audio_path)
        return self.predict(audio)


def list_available_models():
    """List all available trained models"""
    models_dir = Path("./models")
    if not models_dir.exists():
        print("❌ No models directory found")
        return []
    
    model_dirs = []
    for item in models_dir.iterdir():
        if item.is_dir() and "drone_classifier" in item.name:
            if (item / "config.json").exists() and (item / "model.safetensors").exists():
                model_dirs.append(item)
    
    return sorted(model_dirs)


def main():
    """Main prediction interface"""
    parser = argparse.ArgumentParser(description="Predict drone sounds from audio files")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    print("🚁 Drone Sound Prediction System")
    print("=" * 50)
    
    # List available models
    if args.list_models:
        models = list_available_models()
        print(f"📂 Available models ({len(models)}):")
        for i, model in enumerate(models):
            print(f"   {i+1}. {model.name}")
        return
    
    # Get model path
    if args.model:
        model_path = args.model
    else:
        models = list_available_models()
        if not models:
            print("❌ No trained models found")
            return
        
        print(f"📂 Available models:")
        for i, model in enumerate(models):
            print(f"   {i+1}. {model.name}")
        
        choice = input(f"\nSelect model (1-{len(models)}): ")
        try:
            model_path = str(models[int(choice) - 1])
        except (ValueError, IndexError):
            print("❌ Invalid model selection")
            return
    
    # Get audio file path
    if args.audio:
        audio_path = args.audio
    else:
        audio_path = input("\nEnter path to audio file: ")
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    try:
        # Initialize predictor
        predictor = DroneAudioPredictor(model_path)
        
        # Make prediction
        results = predictor.predict_file(audio_path)
        
        # Display results
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(f"📁 Audio File: {audio_path}")
        print(f"🤖 Model: {predictor.model_name}")
        print(f"🎯 Prediction: {results['predicted_label']}")
        print(f"📊 Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
        print(f"🚁 Is Drone: {'YES' if results['is_drone'] else 'NO'}")
        print("\n📈 All Probabilities:")
        for label, prob in results['all_probabilities'].items():
            print(f"   {label}: {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    main()