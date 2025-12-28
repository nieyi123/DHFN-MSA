# DHFN: A Disentangled Hierarchical Fusion Network for Multimodal Sentiment Analysis

Official PyTorch implementation of "DHFN: A Disentangled Hierarchical Fusion Network for Multimodal Sentiment Analysis".

**Authors:** Jiajun Yan, Junfeng Shen  
**Affiliation:** Hubei University, School of Artificial Intelligence

---

## Results

### CMU-MOSI
| Acc-7 | Acc-2 | F1 | Corr | MAE |
|-------|-------|-----|------|-----|
| 46.79 | 85.67 | 85.68 | 0.797 | 0.719 |

### CMU-MOSEI
| Acc-7 | Acc-2 | F1 | Corr | MAE |
|-------|-------|-----|------|-----|
| 52.82 | 85.50 | 85.47 | 0.770 | 0.530 |

---

## Installation
```bash
# Create environment
conda create -n dhfn python=3.8
conda activate dhfn

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

---

## Datasets

Download CMU-MOSI and CMU-MOSEI:
- MOSI: http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
- MOSEI: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/

---

## Training
```bash
# MOSI
python train.py --dataset mosi

# MOSEI
python train.py --dataset mosei
```

---

## Citation
```bibtex
@article{yan2025dhfn,
  title={DHFN: A Disentangled Hierarchical Fusion Network for Multimodal Sentiment Analysis},
  author={Yan, Jiajun and Shen, Junfeng},
  note={Code available at: https://github.com/nieyi123/DHFN-MSA}
}
```

---

## Contact

- Jiajun Yan: junjiayan0205@163.com
- Junfeng Shen: shenjunfengwindy@163.com

---

## License

MIT License
