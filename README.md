# F1TENTH Dashcam to Costmap Translation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning models for translating F1TENTH dashcam images to navigation costmaps using neural networks.

## 🎯 Project Overview

This project implements **image-to-image translation** models that convert dashcam images from F1TENTH autonomous racing cars into costmaps for navigation. The models learn to identify safe and unsafe areas from visual input, enabling autonomous navigation in racing environments.

### Key Features

- **Two Neural Network Architectures**: U-Net and Context Network implementations
- **150×150 Resolution**: Optimized for real-time performance
- **Identity Initialization**: Advanced weight initialization for stable training
- **Morphological Preprocessing**: Enhanced data processing pipeline
- **W&B Integration**: Experiment tracking and visualization

## 🏁 Models

### UNet150
- **Architecture**: Encoder-decoder with skip connections
- **Specialization**: Spatial feature preservation
- **Parameters**: ~270K (complexity_multiplier=4)

### Context Network
- **Architecture**: Dilated convolutions with identity initialization
- **Specialization**: Multi-scale context capture
- **Parameters**: ~47K

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/f1tenth-costmap-translation.git
cd f1tenth-costmap-translation
pip install -r requirements.txt
```

### Data Structure

Organize your data as follows:
```
Data/
├── Dashcams/          # Input dashcam images (150×150)
├── Costmaps/          # Target costmap images (150×150)
└── TrackMap/          # Track maps for costmap generation
```

### Training

Train the U-Net model:
```bash
python scripts/train_unet.py
```

Train the Context Network:
```bash
python scripts/train_context.py
```

### Evaluation

Evaluate trained models and generate visualizations:
```bash
python scripts/evaluate.py
```

### Demo

Explore the interactive demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

## 📊 Results

The models achieve effective costmap generation for F1TENTH navigation:

- **U-Net**: Excellent spatial detail preservation with skip connections
- **Context Network**: Efficient multi-scale context understanding
- **Training Time**: ~15 minutes per epoch on modern GPU
- **Inference Speed**: Real-time capable for autonomous racing

## 🏗️ Project Structure

```
f1tenth-costmap-translation/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── unet.py         # U-Net implementation
│   │   └── context_net.py  # Context Network implementation
│   ├── data/               # Dataset and preprocessing
│   │   ├── dataset.py      # PyTorch dataset classes
│   │   └── preprocessing.py # Data processing utilities
│   └── utils/              # Training utilities
│       └── training.py     # Helper functions
├── scripts/                # Executable scripts
│   ├── train_unet.py      # U-Net training
│   ├── train_context.py   # Context Network training
│   └── evaluate.py        # Model evaluation
├── notebooks/             # Jupyter demonstrations
│   └── demo.ipynb        # Interactive demo
├── models/               # Saved model checkpoints
└── Data/                # Dataset directory
```

## 🔬 Technical Details

### Architecture Highlights

**U-Net150**:
- Encoder: 2 downsampling blocks with MaxPool2d
- Decoder: 2 upsampling blocks with ConvTranspose2d
- Skip connections preserve spatial information
- Double convolution blocks with ReLU activation

**Context Network**:
- Dilated convolutions: [1,1,1,2,4,8,16,32,1] dilation factors
- Identity initialization for training stability
- Progressive channel expansion: [48,64,128,32,32,32,32,32,1]
- Sigmoid output activation

### Training Configuration

- **Loss Function**: L1Loss for sharp boundaries
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 5 (U-Net), 1 (Context Network)
- **Input Normalization**: mean=0.2335, std=0.1712
- **Device**: Automatic CUDA/CPU detection

## 📈 Performance

| Model | Parameters | L1 Loss | Training Speed | Memory Usage |
|-------|------------|---------|----------------|--------------|
| U-Net | ~270K | 0.125 | Fast | Moderate |
| Context Net | ~47K | 0.138 | Very Fast | Low |

## 🔧 Advanced Usage

### Custom Training

Modify hyperparameters in the training scripts:

```python
# In scripts/train_unet.py
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
```

### Model Export

Save models for deployment:

```python
from src.utils import save_model
save_model(model, "path/to/model.pth", epoch=current_epoch)
```

## 🤝 Contributing

This project was developed as part of the AIGS-538 Deep Learning course. The codebase has been refactored for clarity and professional presentation.

## 📜 Citation

```bibtex
@misc{f1tenth2024,
  title={F1TENTH Dashcam to Costmap Translation using Deep Learning},
  author={AIGS-538 Deep Learning Project},
  year={2024},
  note={Image-to-image translation for autonomous racing navigation}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- F1TENTH autonomous racing community
- PyTorch deep learning framework
- Weights & Biases for experiment tracking
- AIGS-538 Deep Learning course at [University Name]

---

⚡ **Ready to race autonomously with computer vision!** 🏎️