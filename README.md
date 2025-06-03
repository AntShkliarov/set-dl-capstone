# 🚁 Drone Sound Recognition - Deep Learning Capstone Project

A state-of-the-art audio classification system for detecting drone sounds using transformer-based deep learning models. This project implements and compares three cutting-edge audio classification architectures: **Wav2Vec2**, **HuBERT**, and **Audio Spectrogram Transformer (AST)**.

## 🏆 Project Highlights

- **Champion Model**: AST achieved **100% accuracy** on drone detection
- **Multi-Model Comparison**: Comprehensive evaluation of 3 transformer architectures
- **Apple Silicon Optimized**: Full MPS acceleration support for M-series chips
- **Production Ready**: Complete pipeline from training to deployment

## 📊 Model Performance

| Model | Accuracy | F1-Score | Demo Performance | 
|-------|----------|----------|------------------|
| **AST** | 🏆 **100.0%** | **100.0%** | 100% (12/12) | 
| **Wav2Vec2** | 99.91% | 99.91% | 100% (12/12) |
| **HuBERT** | 99.82% | 99.82% | 75% (9/12) |

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (for MPS acceleration) or CPU
- 16GB+ RAM recommended

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dl-capstone
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
   ```

## 🚀 Quick Start

### 1. Run Demo Predictions

Test the trained models with pre-recorded audio samples:

```bash
python util/demo_predictions.py
```

### 2. Evaluate All Models

Compare all three models on the test dataset:

```bash
python util/evaluate_models.py
```

### 3. Train a Single Model

Train the champion AST model:

```bash
python src/train/train_ast.py
```

### 4. Full Pipeline Training

Run the complete training pipeline for any model:

```python
from src.drone_pipeline import DroneAudioPipeline

# Train Wav2Vec2 model
pipeline = DroneAudioPipeline("facebook/wav2vec2-base")
model, trainer, results = pipeline.run_full_pipeline()
```

## 📁 Project Structure

```
dl-capstone/
├── src/                           # Core source code
│   ├── drone_pipeline.py          # Main training pipeline
│   ├── dataset/dataset_loader.py  # Dataset handling
│   ├── train/                     # Model-specific training
│   │   ├── train_wav2vec2.py
│   │   ├── train_hubert.py
│   │   └── train_ast.py
│   ├── eval/model_evaluator.py    # Evaluation framework
│   ├── predict/predict_drone_sounds.py  # Inference
│   ├── visual/visualization.py    # Metrics visualization
│   └── test/                      # Test suites
├── models/                        # Trained model artifacts
├── results/                       # Evaluation outputs & visualizations
├── sounds/                        # Test audio samples
├── util/                          # Utility scripts
│   ├── demo_predictions.py        # Interactive demo
│   └── evaluate_models.py         # Batch evaluation
├── notebooks/                     # Jupyter documentation
```

## 🔬 Model Architectures

### 1. **Audio Spectrogram Transformer (AST)** 🏆
- **Architecture**: Vision Transformer adapted for audio spectrograms
- **Pre-training**: AudioSet (2M+ audio clips)
- **Approach**: Treats audio spectrograms as image patches
- **Results**: Perfect 100% accuracy

### 2. **Wav2Vec2**
- **Architecture**: Convolutional feature extraction + Transformer encoder
- **Pre-training**: 960 hours of unlabeled speech (LibriSpeech)
- **Approach**: Self-supervised learning on raw audio waveforms
- **Results**: 99.91% accuracy

### 3. **HuBERT (Hidden-Unit BERT)**
- **Architecture**: BERT-like transformer for speech
- **Pre-training**: LibriSpeech 960h + self-supervised learning
- **Approach**: Masked prediction on discrete speech units
- **Results**: 99.82% accuracy

## 📈 Training Configuration

```python
# Optimized for Apple Silicon
TRAINING_CONFIG = {
    'learning_rate': 3e-5,
    'num_train_epochs': 10,
    'per_device_train_batch_size': 8,
    'weight_decay': 0.01,
    'fp16': False,  # MPS compatibility
    'dataloader_num_workers': 2,
}
```

## 🎯 Usage Examples

### Training a Custom Model

```python
from src.drone_pipeline import DroneAudioPipeline

# Initialize pipeline
pipeline = DroneAudioPipeline("MIT/ast-finetuned-audioset-10-10-0.4593")

# Custom training configuration
custom_config = {
    'num_train_epochs': 5,
    'learning_rate': 5e-5,
    'per_device_train_batch_size': 16
}

# Run complete pipeline
model, trainer, results = pipeline.run_full_pipeline(
    custom_training_args=custom_config
)
```

### Making Predictions

```python
from src.predict.predict_drone_sounds import DroneSoundPredictor

# Load trained model
predictor = DroneSoundPredictor("models/MIT_ast-finetuned-audioset-10-10-0.4593_drone_classifier")

# Predict on audio file
result = predictor.predict_audio("path/to/audio.wav")
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}")
```

### Batch Evaluation

```python
from src.eval.model_evaluator import DroneAudioModelEvaluator

# Evaluate model
evaluator = DroneAudioModelEvaluator("models/model_path", "results/")
results, visualizations = evaluator.run_full_evaluation()

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

## 📊 Evaluation Metrics

The project generates comprehensive evaluation reports including:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrices**: Visual classification performance
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Learning Curves**: Training progression visualization
- **Demo Performance**: Real-world audio sample testing

## 🔧 Technical Features

### Apple Silicon Optimization
- **MPS Backend**: Metal Performance Shaders GPU acceleration
- **Memory Efficiency**: Optimized for unified memory architecture
- **FP32 Precision**: MPS-compatible training precision

### Audio Processing
- **Sample Rate**: 16 kHz standardization
- **Segment Length**: 10-second maximum audio clips
- **Feature Extraction**: Model-specific audio preprocessing
- **Data Augmentation**: Robust audio handling pipeline

### Training Pipeline
- **5-Stage Process**: Load → Pre-Process → Train → Evaluate → Observe
- **Automatic Checkpointing**: Best model preservation
- **Comprehensive Logging**: Training progress tracking
- **Visualization**: Automatic results visualization

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
python -m pytest src/test/

# Run specific test
python -m pytest src/test/model_evaluator.test.py -v
```

## 📋 Dataset

- **Source**: Hugging Face (`geronimobasso/drone-audio-detection-samples`)
- **Classes**: Drone vs Non-Drone sounds
- **Format**: WAV audio files, 16 kHz
- **Split**: Train/Validation/Test with stratification

## 🚀 Production Deployment

### Model Export
```python
# Save model for production
pipeline.trainer.save_model("production_model/")
pipeline.feature_extractor.save_pretrained("production_model/")
```

### Inference API
```python
# Load production model
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model = AutoModelForAudioClassification.from_pretrained("production_model/")
feature_extractor = AutoFeatureExtractor.from_pretrained("production_model/")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This project serves as a capstone demonstration of:
- **Deep Learning**: Transformer architectures for audio
- **Transfer Learning**: Pre-trained model adaptation
- **Audio Processing**: Signal processing and feature extraction
- **Model Evaluation**: Comprehensive performance analysis
- **Production ML**: End-to-end pipeline development

## 🔗 References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [HuBERT Paper](https://arxiv.org/abs/2106.07447)
- [AST Paper](https://arxiv.org/abs/2104.01778)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

**Champion Model**: AST (100% accuracy) ready for production deployment 🏆