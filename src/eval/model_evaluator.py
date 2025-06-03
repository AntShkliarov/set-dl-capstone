#!/usr/bin/env python3
"""
DroneAudioModelEvaluator - A reusable class for evaluating trained audio classification models
"""

import torch
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Any, Optional, Tuple, List
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)
from datasets import Audio

from ..dataset.dataset_loader import DroneDatasetLoader
from ..visual.visualization import AudioClassificationVisualizer


class DroneAudioModelEvaluator:
    """
    A comprehensive evaluator for trained drone audio classification models.
    
    Supports evaluation of different transformer models (Wav2Vec2, HuBERT, AST)
    with automatic feature extraction, preprocessing, evaluation, and visualization.
    """
    
    def __init__(self, model_path: str, results_dir: str = "./results"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model directory
            results_dir: Directory to save results and visualizations
        """
        self.model_path = Path(model_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Model components
        self.feature_extractor = None
        self.model = None
        self.model_name = None
        
        # Data components
        self.dataset = None
        self.eval_dataset = None
        
        # Results
        self.evaluation_results = None
        
        print(f"🔧 Initialized DroneAudioModelEvaluator")
        print(f"   Model path: {self.model_path}")
        print(f"   Results dir: {self.results_dir}")
    
    def load_model_and_feature_extractor(self) -> bool:
        """
        Load the trained model and feature extractor.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                print(f"❌ Model not found at {self.model_path}")
                return False
            
            print("🔄 Loading feature extractor and model...")
            
            # Load feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_path)
            
            # Extract model name from path for identification
            self.model_name = self.model_path.name.replace("_drone_classifier", "")
            
            print(f"✅ Model loaded successfully: {self.model_name}")
            print(f"   Expected sampling rate: {self.feature_extractor.sampling_rate}Hz")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_and_prepare_dataset(self, dataset_name: str = "drone_sampled_0.04") -> bool:
        """
        Load and prepare the dataset for evaluation.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"📂 Loading dataset: {dataset_name}")
            
            # Load dataset
            loader = DroneDatasetLoader()
            self.dataset = loader.load_processed_dataset_safe(dataset_name)
            
            if self.dataset is None:
                print(f"❌ Failed to load dataset: {dataset_name}")
                return False
            
            print(f"✅ Dataset loaded successfully")
            for split_name, split_data in self.dataset.items():
                print(f"   {split_name}: {len(split_data)} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def preprocess_dataset(self, test_size: float = 0.3, eval_size: float = 0.5) -> bool:
        """
        Preprocess the dataset for model evaluation.
        
        Args:
            test_size: Fraction of training data to use for testing
            eval_size: Fraction of test data to use for evaluation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("🔧 Preprocessing dataset...")
            
            # Resample audio to match model's expected sample rate
            for split_name, split_dataset in self.dataset.items():
                self.dataset[split_name] = split_dataset.cast_column(
                    "audio", Audio(sampling_rate=self.feature_extractor.sampling_rate)
                )
            
            # Create evaluation split
            self.eval_dataset = self.dataset["train"].train_test_split(test_size=test_size, seed=42)["test"]
            self.eval_dataset = self.eval_dataset.train_test_split(test_size=eval_size, seed=42)["train"]
            
            # Preprocessing function
            def preprocess_function(examples):
                """Extract features from audio"""
                audio_arrays = [x["array"] for x in examples["audio"]]
                inputs = self.feature_extractor(
                    audio_arrays, 
                    sampling_rate=self.feature_extractor.sampling_rate,
                    max_length=16000 * 10,  # 10 seconds max
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                return inputs
            
            # Apply preprocessing
            self.eval_dataset = self.eval_dataset.map(
                preprocess_function,
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                num_proc=1
            )
            
            print(f"✅ Preprocessing complete: {len(self.eval_dataset)} evaluation samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preprocessing dataset: {e}")
            return False
    
    def evaluate_model(self) -> bool:
        """
        Run model evaluation and calculate predictions.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("📊 Running model evaluation...")
            
            # Create trainer for evaluation
            trainer = Trainer(
                model=self.model,
                tokenizer=self.feature_extractor,
            )
            
            # Get predictions
            predictions = trainer.predict(self.eval_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
            
            # Store predictions
            self.predictions_data = {
                'y_pred': y_pred,
                'y_true': y_true,
                'y_pred_proba': y_pred_proba,
                'raw_predictions': predictions.predictions
            }
            
            print(f"✅ Evaluation complete: {len(y_pred)} predictions generated")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return False
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        print("📊 Calculating metrics...")
        
        y_pred = self.predictions_data['y_pred']
        y_true = self.predictions_data['y_true']
        y_pred_proba = self.predictions_data['y_pred_proba']
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        print(f"✅ Basic Metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        # ROC metrics (for binary classification)
        unique_labels = np.unique(y_true)
        roc_data = {'fpr': None, 'tpr': None, 'auc': None}
        
        if len(unique_labels) == 2 and y_pred_proba.shape[1] == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
                roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}
                print(f"   AUC Score: {auc_score:.4f}")
            except ValueError as e:
                print(f"⚠️  Warning: ROC calculation failed: {e}")
        else:
            print(f"⚠️  Warning: Evaluation set contains only {len(unique_labels)} class(es): {unique_labels}")
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'roc_data': roc_data,
            'predictions': y_pred,
            'true_labels': y_true,
            'prediction_probabilities': y_pred_proba,
            'unique_labels': unique_labels,
            'support': support
        }
        
        self.evaluation_results = results
        return results
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save evaluation results to pickle file.
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path: Path to saved results file
        """
        if filename is None:
            filename = f'{self.model_name.replace("/", "_")}_results.pkl'
        
        results_file = self.results_dir / filename
        
        with open(results_file, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
        
        print(f"💾 Results saved to: {results_file}")
        return results_file
    
    def create_visualizations(self) -> List[Path]:
        """
        Create visualization plots for the evaluation results.
        
        Returns:
            list: List of created visualization file paths
        """
        print("📈 Creating visualizations...")
        
        # Initialize visualizer - delegate all visualization logic to it
        visualizer = AudioClassificationVisualizer(self.results_dir)
        
        # Let visualizer handle training history extraction and visualization creation
        viz_files = visualizer.create_all_visualizations_for_model(
            self.evaluation_results,
            self.model_path,
            self.model_name
        )
        
        created_files = [f for f in viz_files if f is not None and f.exists()]
        print(f"✅ Created {len(created_files)} visualization files")
        
        return created_files
    
    
    def regenerate_learning_curves_only(self) -> Optional[Path]:
        """
        Regenerate only the learning curves for a trained model without full evaluation.
        
        Returns:
            Path: Path to the generated learning curves file, or None if failed
        """
        try:
            print("🔧 Regenerating learning curves only...")
            
            # Extract model name if not already set
            if self.model_name is None:
                self.model_name = self.model_path.name.replace("_drone_classifier", "")
            
            # Delegate to visualizer to handle everything
            visualizer = AudioClassificationVisualizer(self.results_dir)
            curves_file = visualizer.find_and_visualize_model_learning_curves(
                self.model_name, self.model_path.parent
            )
            
            if curves_file:
                print(f"✅ Learning curves regenerated: {curves_file}")
            else:
                print("❌ No training history data found to visualize")
            
            return curves_file
            
        except Exception as e:
            print(f"❌ Error regenerating learning curves: {e}")
            return None
    
    def run_full_evaluation(self, dataset_name: str = "drone_sampled_0.04") -> Tuple[Dict[str, Any], List[Path]]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            dataset_name: Name of the dataset to use for evaluation
            
        Returns:
            tuple: (evaluation_results, visualization_files)
        """
        print("🚀 Starting full evaluation pipeline...")
        print("=" * 60)
        
        # Step 1: Load model
        if not self.load_model_and_feature_extractor():
            raise RuntimeError("Failed to load model")
        
        # Step 2: Load dataset
        if not self.load_and_prepare_dataset(dataset_name):
            raise RuntimeError("Failed to load dataset")
        
        # Step 3: Preprocess data
        if not self.preprocess_dataset():
            raise RuntimeError("Failed to preprocess dataset")
        
        # Step 4: Run evaluation
        if not self.evaluate_model():
            raise RuntimeError("Failed to run evaluation")
        
        # Step 5: Calculate metrics
        results = self.calculate_metrics()
        
        # Step 6: Save results
        results_file = self.save_results()
        
        # Step 7: Create visualizations
        viz_files = self.create_visualizations()
        
        print("\n" + "=" * 60)
        print("EVALUATION PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Results file: {results_file}")
        print(f"Visualizations: {len(viz_files)} files created")
        print("=" * 60)
        
        return results, viz_files
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded model.
        
        Returns:
            dict: Model summary information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "config": self.model.config.to_dict() if hasattr(self.model, 'config') else {},
            "sampling_rate": self.feature_extractor.sampling_rate if self.feature_extractor else None
        }


def evaluate_model(model_path: str, dataset_name: str = "drone_sampled_0.04", results_dir: str = "./results") -> Tuple[Dict[str, Any], List[Path]]:
    """
    Convenience function to evaluate a model with default settings.
    
    Args:
        model_path: Path to the trained model
        dataset_name: Name of the dataset to use
        results_dir: Directory for results
        
    Returns:
        tuple: (evaluation_results, visualization_files)
    """
    evaluator = DroneAudioModelEvaluator(model_path, results_dir)
    return evaluator.run_full_evaluation(dataset_name)


def find_all_trained_models(models_dir: str = "./models") -> List[Path]:
    """
    Find all trained models in the models directory.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        list: List of model directory paths
    """
    models_dir = Path(models_dir)
    
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


def find_latest_checkpoint(model_path: Path) -> Path:
    """
    Find the latest checkpoint for a model, if available.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Path: Path to the latest checkpoint or original model path
    """
    checkpoints = list(model_path.glob("checkpoint-*"))
    if checkpoints:
        # Use the latest checkpoint (highest number)
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split('-')[1]))[-1]
        print(f"📌 Using latest checkpoint: {latest_checkpoint.name}")
        return latest_checkpoint
    else:
        print("📌 Using base model (no checkpoints found)")
        return model_path


def evaluate_all_models(models_dir: str = "./models", dataset_name: str = "drone_sampled_0.04", results_dir: str = "./results") -> List[Dict[str, Any]]:
    """
    Evaluate all available trained models.
    
    Args:
        models_dir: Directory containing trained models
        dataset_name: Dataset to use for evaluation
        results_dir: Directory for results
        
    Returns:
        list: List of evaluation results for all models
    """
    print("🎯 Evaluating All Available Models")
    print("=" * 50)
    
    models = find_all_trained_models(models_dir)
    
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
            # Use latest checkpoint if available
            actual_model_path = find_latest_checkpoint(model_path)
            
            # Evaluate using DroneAudioModelEvaluator
            evaluator = DroneAudioModelEvaluator(str(actual_model_path), results_dir)
            results, viz_files = evaluator.run_full_evaluation(dataset_name)
            
            all_results.append({
                'model_path': str(model_path),
                'actual_model_path': str(actual_model_path),
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


def compare_model_performance(all_results: List[Dict[str, Any]]) -> None:
    """
    Compare performance across all evaluated models.
    
    Args:
        all_results: List of evaluation results from evaluate_all_models
    """
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