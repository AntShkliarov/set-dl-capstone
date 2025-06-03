"""
Visualization utilities for audio classification results
Separated from main classifier for better modularity
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path


class AudioClassificationVisualizer:
    """Handles all visualization tasks for audio classification results"""
    
    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set consistent style
        plt.style.use('seaborn-v0_8')
    
    def plot_confusion_matrix(self, confusion_matrix, model_name, class_names=None):
        """Create confusion matrix visualization"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = self.results_dir / f'{model_name.replace("/", "_")}_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_roc_curve(self, fpr, tpr, auc_score, model_name):
        """Create ROC curve visualization"""
        # Validate data before plotting
        if len(fpr) == 0 or len(tpr) == 0 or len(fpr) != len(tpr):
            print(f"⚠️  Warning: Invalid ROC data for {model_name} (fpr: {len(fpr)}, tpr: {len(tpr)})")
            print("   Skipping ROC curve visualization")
            return None
            
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        filename = self.results_dir / f'{model_name.replace("/", "_")}_roc_curve.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_learning_curves(self, training_history, model_name):
        """Create comprehensive learning curves"""
        if not training_history['train_loss']:
            print("No training history available for learning curves")
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
        if training_history['eval_loss']:
            ax1.plot(epochs[:len(training_history['eval_loss'])], 
                    training_history['eval_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        if training_history['eval_accuracy']:
            ax2.plot(epochs[:len(training_history['eval_accuracy'])], 
                    training_history['eval_accuracy'], 'g-', label='Validation Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        # F1 Score curve
        if training_history['eval_f1']:
            ax3.plot(epochs[:len(training_history['eval_f1'])], 
                    training_history['eval_f1'], 'm-', label='Validation F1')
            ax3.set_title('Validation F1 Score')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('F1 Score')
            ax3.legend()
            ax3.grid(True)
        
        # Precision and Recall curves
        if training_history['eval_precision'] and training_history['eval_recall']:
            ax4.plot(epochs[:len(training_history['eval_precision'])], 
                    training_history['eval_precision'], 'c-', label='Validation Precision')
            ax4.plot(epochs[:len(training_history['eval_recall'])], 
                    training_history['eval_recall'], 'y-', label='Validation Recall')
            ax4.set_title('Validation Precision and Recall')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        filename = self.results_dir / f'{model_name.replace("/", "_")}_learning_curves.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_all_visualizations(self, results, training_history, model_name):
        """Create all visualizations for a model's results"""
        print("📈 Creating visualizations...")
        
        created_files = []
        
        # Confusion Matrix
        cm_file = self.plot_confusion_matrix(
            results['confusion_matrix'],
            model_name,
            class_names=['No Drone', 'Drone']  # Default binary classes
        )
        created_files.append(cm_file)
        
        # ROC Curve (if binary classification)
        if results['roc_data']['fpr'] is not None:
            roc_file = self.plot_roc_curve(
                results['roc_data']['fpr'],
                results['roc_data']['tpr'],
                results['roc_data']['auc'],
                model_name
            )
            if roc_file is not None:
                created_files.append(roc_file)
        
        # Learning Curves
        learning_file = self.plot_learning_curves(training_history, model_name)
        if learning_file:
            created_files.append(learning_file)
        
        print(f"✅ Created {len(created_files)} visualization files")
        return created_files

    def create_all_visualizations_for_model(self, results, model_path, model_name):
        """Create all visualizations for a model including automatic training history extraction"""
        print("📈 Creating visualizations...")
        
        created_files = []
        
        # Confusion Matrix
        cm_file = self.plot_confusion_matrix(
            results['confusion_matrix'],
            model_name,
            class_names=['No Drone', 'Drone']  # Default binary classes
        )
        created_files.append(cm_file)
        
        # ROC Curve (if binary classification)
        if results['roc_data']['fpr'] is not None:
            roc_file = self.plot_roc_curve(
                results['roc_data']['fpr'],
                results['roc_data']['tpr'],
                results['roc_data']['auc'],
                model_name
            )
            if roc_file is not None:
                created_files.append(roc_file)
        
        # Learning Curves - automatically extract training history
        learning_file = self.find_and_visualize_model_learning_curves(model_name, model_path.parent)
        if learning_file:
            created_files.append(learning_file)
        
        print(f"✅ Created {len(created_files)} visualization files")
        return created_files

    def extract_training_history_from_trainer_state(self, trainer_state_path):
        """Extract training history from HuggingFace trainer state JSON"""
        
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        log_history = trainer_state['log_history']
        
        # Initialize training history structure
        training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': [],
            'eval_precision': [],
            'eval_recall': []
        }
        
        print(f"📊 Extracting training data from {len(log_history)} log entries...")
        
        # Process log history
        for entry in log_history:
            # Training loss (logged at each step)
            if 'loss' in entry:
                training_history['train_loss'].append(entry['loss'])
            
            # Evaluation metrics (logged at each epoch)
            if 'eval_loss' in entry:
                training_history['eval_loss'].append(entry['eval_loss'])
            if 'eval_accuracy' in entry:
                training_history['eval_accuracy'].append(entry['eval_accuracy'])
            if 'eval_f1' in entry:
                training_history['eval_f1'].append(entry['eval_f1'])
            if 'eval_precision' in entry:
                training_history['eval_precision'].append(entry['eval_precision'])
            if 'eval_recall' in entry:
                training_history['eval_recall'].append(entry['eval_recall'])
        
        # Print summary
        print(f"✅ Extracted training history:")
        print(f"   📈 Training loss: {len(training_history['train_loss'])} data points")
        print(f"   📊 Evaluation loss: {len(training_history['eval_loss'])} epochs")
        print(f"   🎯 Accuracy: {len(training_history['eval_accuracy'])} epochs")
        print(f"   📏 F1 Score: {len(training_history['eval_f1'])} epochs")
        print(f"   🎯 Precision: {len(training_history['eval_precision'])} epochs")
        print(f"   🎯 Recall: {len(training_history['eval_recall'])} epochs")
        
        return training_history

    def visualize_learning_curves_from_trainer_state(self, trainer_state_path, model_name):
        """Create learning curves from HuggingFace trainer state file"""
        
        trainer_state_path = Path(trainer_state_path)
        if not trainer_state_path.exists():
            print(f"❌ Trainer state file not found: {trainer_state_path}")
            return None
        
        print(f"📂 Reading trainer state from: {trainer_state_path}")
        
        # Extract training history
        training_history = self.extract_training_history_from_trainer_state(trainer_state_path)
        
        # Generate learning curves
        print("\n🎨 Creating learning curves...")
        learning_curves_file = self.plot_learning_curves(training_history, model_name)
        
        if learning_curves_file:
            print(f"✅ Learning curves saved to: {learning_curves_file}")
            
            # Print final training metrics
            print(f"\n📊 Final Training Results for {model_name}:")
            if training_history['eval_accuracy']:
                final_accuracy = training_history['eval_accuracy'][-1]
                print(f"   🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
                
            if training_history['eval_f1']:
                final_f1 = training_history['eval_f1'][-1]
                print(f"   📏 Final F1 Score: {final_f1:.4f}")
                
            if training_history['eval_loss']:
                final_eval_loss = training_history['eval_loss'][-1]
                print(f"   📉 Final Validation Loss: {final_eval_loss:.6f}")
                
            if training_history['train_loss']:
                final_train_loss = training_history['train_loss'][-1]
                print(f"   📈 Final Training Loss: {final_train_loss:.6f}")
        
        return learning_curves_file

    def find_and_visualize_model_learning_curves(self, model_name, models_dir="./models"):
        """
        Automatically find trainer state for a model and create learning curves
        
        Args:
            model_name: Name of the model (e.g., "facebook/wav2vec2-base", "facebook/hubert-base-ls960")
            models_dir: Directory containing model checkpoints
        """
        models_dir = Path(models_dir)
        model_dir_name = f"{model_name.replace('/', '_')}_drone_classifier"
        model_dir = models_dir / model_dir_name
        
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return None
        
        # Find the latest checkpoint with trainer_state.json
        trainer_state_files = list(model_dir.glob("**/trainer_state.json"))
        
        if not trainer_state_files:
            print(f"❌ No trainer_state.json files found in {model_dir}")
            return None
        
        # Use the most recent trainer state (highest checkpoint number)
        latest_trainer_state = max(trainer_state_files,
                                 key=lambda p: int(p.parent.name.split('-')[-1]) if 'checkpoint-' in p.parent.name else 0)
        
        print(f"🔍 Found trainer state: {latest_trainer_state}")
        
        # Create learning curves
        return self.visualize_learning_curves_from_trainer_state(latest_trainer_state, model_name)