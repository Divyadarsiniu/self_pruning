# self_pruning
# 🧠 Self-Pruning Neural Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A neural network that **learns to prune itself** during training - eliminating unimportant connections automatically, without post-training optimization.

## 🎯 Key Innovation

Traditional neural networks are trained first, then pruned. This network **prunes itself during training** by learning which connections are important vs unnecessary. Each weight has a learnable "gate" that determines its survival.

## 📊 Results at a Glance

| Configuration | Test Accuracy | Sparsity | Compression |
|---------------|--------------|----------|-------------|
| λ = 0.0001 (Low penalty) | 54.2% | 12% | 1.14x |
| **λ = 0.001 (Recommended)** | **52.9%** | **39%** | **1.64x** |
| λ = 0.01 (High penalty) | 48.2% | 71% | 3.48x |

**Best trade-off:** 39% sparsity with only 1.3% accuracy loss!

## 🌟 Features

- ✅ **Self-pruning mechanism** - Network learns which weights to keep
- ✅ **Adaptive sparsity scheduling** - λ increases during training
- ✅ **Temperature annealing** - Soft → hard gates for better convergence
- ✅ **L1 regularization** - Creates exact zeros (true pruning)
- ✅ **Comprehensive visualization** - Gate distributions, training curves
- ✅ **Production ready** - Easy to train and evaluate


<img width="1782" height="580" alt="gate_distribution_0 0001" src="https://github.com/user-attachments/assets/1ce9c5cd-3a81-4a61-9023-4a818e19c2ea" />


## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install torch torchvision matplotlib numpy pandas tqdm
<img width="1782" height="580" alt="gate_distribution_0 0001" src="https://github.com/user-attachments/assets/1bc87431-e135-45c0-9e17-23a31faec657" />
