# AI-Generated Video Detection

A deep learning model for detecting AI-generated videos with 82% accuracy, trained on a dataset of 20,000 videos.

## Overview

This project implements a neural network architecture for distinguishing between real and AI-generated videos. The model processes videos through multiple stages of feature extraction and classification, utilizing both spatial and temporal information.

## Architecture

The model consists of three main components:

1. **Latent Encoder** (`FullLatentEncoder`)
   - Processes raw video frames through 3 convolutional layers
   - Reduces spatial dimensions by a factor of 8
   - Uses batch normalization for training stability
   - Output channels: 32 → 64 → 128

2. **Patch Encoder** (`FullPatchEncoder`)
   - Extracts 8x8 patches from latent representations
   - Processes patches through convolutional layers
   - Maps features to 768-dimensional embeddings
   - Uses layer normalization for better training stability

3. **Classifier** (`FullClassifier`)
   - Transformer-based architecture with 12 layers
   - Self-attention mechanism for temporal relationships
   - Final classification into real/fake categories

## Performance

- **Accuracy**: 82% on test set
- **Training Data**: 20,000 videos (balanced between real and AI-generated)
- **Hardware Requirements**: 
  - Minimum: 12GB GPU VRAM for inference
  - Recommended: 20GB+ GPU VRAM for training

## Usage

### Training

```bash
python full_train.py
```

The training script includes:
- Memory optimization
- TensorBoard logging
- Checkpoint saving

### Inference and Feature Importance

```bash
python interpret.py
```

The interpretation script provides:
- Integrated Gradients visualization
- Frame-by-frame attribution analysis
- Feature importance heatmaps
- Video output with highlighted regions

## File Structure

- `full_train.py`: Main training script
- `full_scale_classifier.py`: Model architecture implementation
- `dataset.py`: Data loading and preprocessing
- `interpret.py`: Model interpretation and visualization
- `requirements.txt`: Required dependencies

## Dependencies

- PyTorch
- OpenCV
- NumPy
- TensorBoard
- Captum (for interpretability)
- scikit-learn

## Visualization

The project includes comprehensive visualization tools:
- Training metrics plots
- Confusion matrices
- ROC curves
- Feature attribution heatmaps
- Per-class performance metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{aigenvideodetection2024,
  title={AI-Generated Video Detection Using Deep Learning},
  author={Joshua Weg),
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/Joshuaweg/AIgendetector}}
}
``` 
