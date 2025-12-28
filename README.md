# DHFN: A Disentangled Hierarchical Fusion Network for Multimodal Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Pattern%20Analysis%20and%20Applications-green.svg)](https://link.springer.com/journal/10044)

Official PyTorch implementation of **"DHFN: A Disentangled Hierarchical Fusion Network for Multimodal Sentiment Analysis"** published in *Pattern Analysis and Applications*, 2025.

**Authors:** Jiajun Yan, Junfeng Shen  
**Affiliation:** School of Artificial Intelligence, Hubei University, Wuhan, China

---

## 🔥 Highlights

- **Explicit Feature Disentanglement:** Separates modality-specific and modality-shared features to reduce cross-modal interference
- **Hierarchical Learning Module:** Captures multi-level complementary information through learnable hypergraph tokens
- **Dual-Stream Architecture:** Differentiated processing pathways tailored for disentangled feature types
- **State-of-the-Art Performance:** Achieves competitive results on CMU-MOSI and CMU-MOSEI benchmarks

---

## 📊 Main Results

### CMU-MOSI Dataset
| Metric | Acc-7 (%) | Acc-2 (%) | F1 (%) | Corr | MAE |
|--------|-----------|-----------|--------|------|-----|
| **DHFN** | **46.79** | **85.67** | 85.68 | **0.797** | **0.719** |

### CMU-MOSEI Dataset
| Metric | Acc-7 (%) | Acc-2 (%) | F1 (%) | Corr | MAE |
|--------|-----------|-----------|--------|------|-----|
| **DHFN** | 52.82 | 85.50 | 85.47 | **0.770** | **0.530** |

**Key advantages:**
- Best MAE and Correlation on both datasets
- Near-zero cross-polarity errors (no confusion between strongly positive and strongly negative)
- Robust performance on extreme sentiment classes

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM recommended

### Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/nieyi123/DHFN-MSA.git
cd DHFN-MSA
```

#### Step 2: Create Conda Environment
```bash
conda create -n dhfn python=3.8
conda activate dhfn
```

#### Step 3: Install PyTorch
**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 📦 Datasets

### Download CMU-MOSI and CMU-MOSEI

**CMU-MOSI:**
- Official link: http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
- Contains 2,199 video clips from 93 YouTube speakers
- Sentiment labels: continuous [-3, +3]

**CMU-MOSEI:**
- Official link: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
- Contains 22,856 video clips from 1,000+ speakers
- Sentiment labels: continuous [-3, +3]

### Data Preparation

Place the downloaded datasets in the following structure:
```
data/
├── mosi/
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
└── mosei/
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

**Note:** The unaligned version is used by default (text, audio, and visual modalities have different sequence lengths).

---

## 🏋️ Training

### Train on CMU-MOSI
```bash
python train.py --dataset mosi --config configs/mosi.yaml
```

### Train on CMU-MOSEI
```bash
python train.py --dataset mosei --config configs/mosei.yaml
```

### Key Training Arguments
```bash
--dataset        # Dataset name: mosi or mosei
--learning_rate  # Learning rate (default: 1e-4)
--batch_size     # Batch size (default: 16)
--epochs         # Number of epochs (default: 100)
--early_stop     # Early stopping patience (default: 10)
--gpu            # GPU device ID (default: 0)
```

### Monitor Training
Training logs and checkpoints are saved in:
```
./pt/               # Model checkpoints
./logs/             # Training logs
```

---

## 🧪 Evaluation

### Test on CMU-MOSI
```bash
python test.py --dataset mosi --checkpoint pt/DHFN_mosi.pth
```

### Test on CMU-MOSEI
```bash
python test.py --dataset mosei --checkpoint pt/DHFN_mosei.pth
```

### Evaluation Metrics
The model reports the following metrics:
- **Acc-7:** 7-class accuracy (sentiment intensity classification)
- **Acc-2:** Binary accuracy (positive/negative)
- **F1:** Binary F1-score
- **Corr:** Pearson correlation coefficient
- **MAE:** Mean absolute error

---

## 🏗️ Model Architecture

DHFN consists of four core modules:

### 1. Feature Extraction Module
- **Text:** BERT-base-uncased (768-dim)
- **Audio:** 1D convolution for temporal modeling (74-dim)
- **Visual:** 1D convolution for temporal modeling (35-dim)

### 2. Feature Disentanglement Module
- **Modality-Specific Encoders:** Three independent Transformer encoders
- **Modality-Shared Encoder:** Single Transformer encoder shared across modalities
- **Reconstruction Decoders:** Ensure information completeness

### 3. Dual-Stream Enhancement Module
- **Hierarchical Learning (Specific Features):** Multi-level cross-modal fusion with hypergraph tokens
- **Self-Attention Enhancement (Shared Features):** Reinforce cross-modal consistency

### 4. Fusion and Prediction Module
- Integrates enhanced features from both streams
- Multi-level supervision strategy

**Total Parameters:** ~110M

---

## 📁 Project Structure
```
DHFN-MSA/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
│
├── models/                # Model implementations
│   ├── __init__.py
│   ├── dhfn.py           # Main DHFN model
│   ├── encoders.py       # Encoder modules
│   ├── hierarchical.py   # Hierarchical learning module
│   └── losses.py         # Loss functions
│
├── data/                  # Data processing
│   ├── __init__.py
│   └── dataloader.py     # Dataset loader
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── metrics.py        # Evaluation metrics
│   └── helpers.py        # Helper functions
│
├── configs/               # Configuration files
│   ├── mosi.yaml         # MOSI hyperparameters
│   └── mosei.yaml        # MOSEI hyperparameters
│
├── train.py              # Training script
├── test.py               # Evaluation script
└── pt/                   # Model checkpoints (created during training)
```

---

## 🔧 Ablation Studies

To run ablation experiments, modify the configuration:
```python
# Disable feature disentanglement
args.use_disentanglement = False

# Disable hierarchical learning
args.use_hierarchical_learning = False

# Disable shared feature enhancement
args.use_shared_enhance = False
```

Results are reported in the paper (Section 4.5).

---

## 🐛 Troubleshooting

### Installation Issues

**CUDA version mismatch:**
```bash
# Check your CUDA version
nvidia-smi

# Install corresponding PyTorch version
# Visit: https://pytorch.org/get-started/locally/
```

**Transformers version conflict:**
```bash
pip install transformers==4.33.1  # Use exact version if needed
```

**Out of memory during training:**
```bash
# Reduce batch size in configs/mosi.yaml
batch_size: 8  # Default is 16
```

### Runtime Issues

**FileNotFoundError for datasets:**
- Ensure datasets are placed in `data/mosi/` or `data/mosei/`
- Check file names: `train.pkl`, `valid.pkl`, `test.pkl`

**CUDA out of memory:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

---

## 📝 Citation

If you find this code useful for your research, please cite our paper:
```bibtex
@article{yan2025dhfn,
  title={DHFN: A Disentangled Hierarchical Fusion Network for Multimodal Sentiment Analysis},
  author={Yan, Jiajun and Shen, Junfeng},
  journal={Pattern Analysis and Applications},
  year={2025},
  publisher={Springer}
}
```

---

## 📧 Contact

For questions, issues, or collaboration opportunities:

- **Jiajun Yan:** junjiayan0205@163.com
- **Junfeng Shen (Corresponding Author):** shenjunfengwindy@163.com

**Institution:** School of Artificial Intelligence, Hubei University, Wuhan, 430062, Hubei, China

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the creators of the CMU-MOSI and CMU-MOSEI datasets for making their data publicly available:

- Zadeh, A., et al. (2016). "Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Verbal Messages." *IEEE Intelligent Systems*.
- Zadeh, A., et al. (2018). "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph." *ACL*.

We also acknowledge the open-source community for PyTorch, Transformers, and other libraries that made this work possible.

---

## 📊 Additional Resources

- **Paper (arXiv):** [Coming soon]
- **Supplementary Materials:** [Coming soon]
- **Pretrained Models:** [Coming soon]

---

**Last Updated:** December 2024  
**Maintained by:** Jiajun Yan ([@nieyi123](https://github.com/nieyi123))
