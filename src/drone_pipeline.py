"""
Drone Audio Classification Pipeline
Optimized structure: Load -> Pre-Process -> Train -> Evaluate -> Observe
"""

import torch
import numpy as np
from datasets import Audio
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support
)
from pathlib import Path

from .visual.visualization import AudioClassificationVisualizer
from .eval.model_evaluator import DroneAudioModelEvaluator


class DroneAudioPipeline:
    """Optimized drone audio classification pipeline with clear stages"""
    
    def __init__(self, model_name="facebook/wav2vec2-base", output_dir="./models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.results_dir = Path("./results")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = AudioClassificationVisualizer(self.results_dir)
        
        # Training history tracking
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': [],
            'eval_precision': [],
            'eval_recall': []
        }
        
        # Pipeline state
        self.dataset = None
        self.processed_dataset = None
        self.feature_extractor = None
        self.model = None
        self.trainer = None
        self.results = None

    # STAGE 1: LOAD
    def load_data(self, dataset_name=None):
        """Load drone audio dataset"""
        print("🔄 STAGE 1: Loading dataset...")
        
        from .dataset.dataset_loader import DroneDatasetLoader
        loader = DroneDatasetLoader()
        
        # Load dataset with proper validation and structure normalization
        if dataset_name:
            self.dataset = loader.load_processed_dataset_safe(dataset_name)
        else:
            self.dataset = loader.load_raw_dataset_safe()
        
        sample_count = loader.get_sample_count(self.dataset)
        print(f"✅ Dataset loaded: {sample_count} samples")
        return self.dataset

    # STAGE 2: PRE-PROCESS
    def preprocess_data(self):
        """Prepare dataset with feature extraction and preprocessing"""
        print("🔧 STAGE 2: Preprocessing dataset...")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        # Load feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        
        # Resample audio to match model's expected sample rate
        if isinstance(self.dataset, dict):
            for split_name, split_dataset in self.dataset.items():
                self.dataset[split_name] = split_dataset.cast_column(
                    "audio", Audio(sampling_rate=self.feature_extractor.sampling_rate)
                )
        else:
            self.dataset = self.dataset.cast_column(
                "audio", Audio(sampling_rate=self.feature_extractor.sampling_rate)
            )
        
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
        if isinstance(self.dataset, dict):
            for split_name, split_dataset in self.dataset.items():
                self.dataset[split_name] = split_dataset.map(
                    preprocess_function,
                    remove_columns=["audio"],
                    batched=True,
                    batch_size=100,
                    num_proc=1
                )
        else:
            self.dataset = self.dataset.map(
                preprocess_function,
                remove_columns=["audio"],
                batched=True,
                batch_size=100,
                num_proc=1
            )
        
        self.processed_dataset = self.dataset
        print("✅ Dataset preprocessing completed")
        return self.processed_dataset

    def setup_model(self):
        """Setup model configuration"""
        print("🤖 Setting up model...")
        
        if self.processed_dataset is None:
            raise ValueError("Dataset not preprocessed. Call preprocess_data() first.")
        
        # Determine labels
        labels = self.processed_dataset["train"].features.get("label", None)
        if labels is not None:
            num_labels = len(labels.names) if hasattr(labels, 'names') else 2
            id2label = {i: f"label_{i}" for i in range(num_labels)}
            label2id = {v: k for k, v in id2label.items()}
        else:
            num_labels = 2
            id2label = {0: "no_drone", 1: "drone"}
            label2id = {"no_drone": 0, "drone": 1}
        
        # Load model with special handling for AST models
        if "ast" in self.model_name.lower():
            # For AST model, we need to ignore classifier head mismatches as it has been pre-trained on a different number of classes
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True  # Ignore classifier head size mismatch
            )
            print("🔧 AST model loaded with classifier head adaptation")
        else:

            # For other models (Wav2Vec2, HuBERT), use standard loading
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
        
        print(f"✅ Model setup completed: {num_labels} classes")
        return self.model, num_labels, id2label, label2id

    def prepare_data_splits(self):
        """Create train/validation/test splits"""
        if "validation" not in self.processed_dataset:
            dataset_split = self.processed_dataset["train"].train_test_split(test_size=0.3, seed=42)
            train_dataset = dataset_split["train"]
            temp_dataset = dataset_split["test"].train_test_split(test_size=0.5, seed=42)
            eval_dataset = temp_dataset["train"]
            test_dataset = temp_dataset["test"]
        else:
            train_dataset = self.processed_dataset["train"]
            eval_dataset = self.processed_dataset["validation"]
            test_dataset = self.processed_dataset.get("test", eval_dataset)
        
        print(f"Dataset splits: Train={len(train_dataset)}, Val={len(eval_dataset)}, Test={len(test_dataset)}")
        return train_dataset, eval_dataset, test_dataset

    def _get_device_info(self):
        """Get device information and memory constraints"""
        if torch.backends.mps.is_available():
            device = "mps"
            # MPS doesn't have direct memory query, estimate based on system
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            device = "cpu"
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return {
            'device': device,
            'memory': memory_gb,
            'has_mixed_precision': device in ['cuda', 'mps']
        }

    def _get_model_output_dir(self):
        """Generate consistent model output directory path"""
        return self.output_dir / f"{self.model_name.replace('/', '_')}_drone_classifier"

    def _get_training_config(self, device_info, custom_config=None):
        """Generate optimized training configuration based on device capabilities"""
        # Base configuration with reasonable defaults
        base_config = {
            'learning_rate': 3e-5,
            'num_train_epochs': 10,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'save_total_limit': 2,
            'logging_steps': 10,
            'eval_strategy': "epoch",
            'save_strategy': "epoch",
        }
        
        # Device-specific optimizations
        if device_info['device'] == 'mps':
            base_config.update({
                'fp16': False,  # MPS has issues with fp16
                'dataloader_num_workers': 2,
                'per_device_train_batch_size': 12 if device_info['memory'] > 16 else 8,
            })
        else:  # CPU
            base_config.update({
                'fp16': False,
                'dataloader_num_workers': 2,
                'per_device_train_batch_size': 4,  # Smaller for CPU
            })
        
        # Apply custom configuration overrides
        if custom_config:
            base_config.update(custom_config)
        
        # Create TrainingArguments with validated paths
        return TrainingArguments(
            output_dir=str(self._get_model_output_dir()),
            logging_dir=str(self.results_dir / "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            **base_config
        )

    # STAGE 3: TRAIN
    def train_model(self, train_dataset, eval_dataset, custom_training_args=None):
        """Train the model with optional custom training arguments"""
        print("🏋️ STAGE 3: Training model...")
        
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        # Use existing device info and training config methods
        device_info = self._get_device_info()
        print(f"Using device: {device_info['device']}")
        
        # Use existing training config method
        training_args = self._get_training_config(device_info, custom_training_args)
        
        if custom_training_args:
            print(f"🔧 Custom training config applied: {custom_training_args}")
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.feature_extractor,
            compute_metrics=self._compute_metrics,
        )
        
        # Add training history callback
        self._add_training_callback()
        
        # Train
        self.trainer.train()
        
        # Save model
        model_output_dir = self._get_model_output_dir()
        self.trainer.save_model()
        self.feature_extractor.save_pretrained(str(model_output_dir))
        
        print("✅ Training completed")
        return self.trainer

    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def _add_training_callback(self):
        """Add callback to track training history"""
        class TrainingHistoryCallback(TrainerCallback):
            def __init__(self, pipeline):
                super().__init__()
                self.pipeline = pipeline
            
            def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
                if logs:
                    self.pipeline.training_history['eval_loss'].append(logs.get('eval_loss', 0))
                    self.pipeline.training_history['eval_accuracy'].append(logs.get('eval_accuracy', 0))
                    self.pipeline.training_history['eval_f1'].append(logs.get('eval_f1', 0))
                    self.pipeline.training_history['eval_precision'].append(logs.get('eval_precision', 0))
                    self.pipeline.training_history['eval_recall'].append(logs.get('eval_recall', 0))
            
            def on_log(self, args, state, control, model, logs=None, **kwargs):
                if logs and 'train_loss' in logs:
                    self.pipeline.training_history['train_loss'].append(logs['train_loss'])
        
        self.trainer.add_callback(TrainingHistoryCallback(self))

    # STAGE 4: EVALUATE
    def evaluate_model(self, dataset_name=None):
        """STAGE 4: Comprehensive model evaluation using DroneAudioModelEvaluator"""
        print("📊 STAGE 4: Evaluating model...")
        
        if self.trainer is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Use DroneAudioModelEvaluator for comprehensive evaluation
        try:
            model_path = self._get_model_output_dir()
            evaluator = DroneAudioModelEvaluator(str(model_path), str(self.results_dir))
            # Use dataset_name if provided, otherwise evaluator will use its default
            if dataset_name:
                self.results, viz_files = evaluator.run_full_evaluation(dataset_name)
            else:
                self.results, viz_files = evaluator.run_full_evaluation()
            
            print("✅ Evaluation completed using DroneAudioModelEvaluator")
            return self.results, viz_files
            
        except Exception as e:
            print(f"❌ DroneAudioModelEvaluator failed: {e}")
            print("📌 Falling back to basic evaluation...")
            return self._basic_evaluate_fallback()

    def _basic_evaluate_fallback(self):
        """Simple fallback evaluation if DroneAudioModelEvaluator fails"""
        # Create a simple evaluation dataset from processed data
        _, eval_dataset, _ = self.prepare_data_splits()
        
        # Get basic predictions
        predictions = self.trainer.predict(eval_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Basic metrics only
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        return self.results, []
    

    # STAGE 5: OBSERVE
    def observe_results(self):
        """Create visualizations and analyze results"""
        print("👁️ STAGE 5: Observing results...")
        
        if self.results is None:
            raise ValueError("No results to observe. Call evaluate_model() first.")
        
        # Create visualizations
        viz_files = self.visualizer.create_all_visualizations(
            self.results, 
            self.training_history, 
            self.model_name
        )
        
        # Print summary
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(f"Model: {self.model_name}")
        print(f"Accuracy: {self.results['accuracy']:.4f}")
        print(f"Precision: {self.results['precision']:.4f}")
        print(f"Recall: {self.results['recall']:.4f}")
        print(f"F1 Score: {self.results['f1_score']:.4f}")
        if self.results['roc_data']['auc']:
            print(f"AUC Score: {self.results['roc_data']['auc']:.4f}")
        print(f"Visualizations saved: {len(viz_files)} files")
        print("="*50)
        
        return viz_files

    def run_full_pipeline(self, dataset_name=None, custom_training_args=None):
        """
        Execute the complete pipeline: Load -> Pre-Process -> Train -> Evaluate -> Observe
        
        Args:
            dataset_name: Name of dataset to load
            custom_training_args: Custom training configuration
        """
        print("🚀 Starting Full Audio Classification Pipeline")
        print("=" * 60)
        
        try:
            # STAGE 1: Load
            self.load_data(dataset_name)
            
            # STAGE 2: Pre-Process
            self.preprocess_data()
            self.setup_model()
            train_dataset, eval_dataset, test_dataset = self.prepare_data_splits()
            
            # STAGE 3: Train
            self.train_model(train_dataset, eval_dataset, custom_training_args)
            
            # STAGE 4: Evaluate - Now uses DroneAudioModelEvaluator by default
            self.results, viz_files = self.evaluate_model(dataset_name)
            
            # STAGE 5: Observe (only if no visualizations were created)
            if not viz_files:
                self.observe_results()
            
            print("\n🎉 Pipeline completed successfully!")
            return self.model, self.trainer, self.results
            
        except Exception as e:
            print(f"❌ Pipeline failed at current stage: {e}")
            raise