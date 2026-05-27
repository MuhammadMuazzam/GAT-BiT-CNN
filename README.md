# GAC-BiT-CNN: Enhancing Functional Prediction of Methyltransferases Using a Hybrid Deep Learning Model with Protein Language Model Embeddings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This repository contains the complete source code, datasets, and trained model files for the paper:

> **"Enhancing Functional Prediction of Methyltransferases Using a Hybrid GAC-BiT-CNN Deep Learning Model with Protein Language Model Embeddings"**
> Muhammad Muazzam, Farman Ali, Muhammad Asif
> Department of Computer Science, Bahria University Islamabad, Pakistan

GAC-BiT-CNN is a novel hybrid deep learning architecture that integrates:
- **G**enerative **A**dversarial **C**apsule Networks (GAC)
- **Bi**directional **T**ransformers (BiT)
- **C**onvolutional **N**eural **N**etworks (CNN)

Combined with four state-of-the-art protein language model embeddings:
- **ProtGen** (1024-dim)
- **ESM-2** (1280-dim)
- **ProtT5-BFD** (1024-dim)
- **ProtT5-XL-UniRef50** (1024-dim)
- **Fused** (4352-dim)

---

## 📊 Results Summary

| Dataset        | Accuracy | Sensitivity | Specificity | AUC    | MCC   |
|---------------|----------|-------------|-------------|--------|-------|
| Training (CV)  | 98.34%   | 98.39%      | 98.29%      | 99.78% | 0.967 |
| Independent Test | 98.16% | 97.89%      | 98.42%      | 99.74% | 0.963 |

---

## 📁 Repository Structure

```
GAC-BiT-CNN-MT/
│
├── data/
│   ├── data_training.txt          # Training dataset (FASTA format) — 2290 PMT + 2290 non-PMT
│   ├── data_testing.txt           # Independent test dataset (FASTA format) — 572 PMT + 572 non-PMT
│   └── external_dataset.txt       # External/blind validation dataset (FASTA format)
│
├── code/
│   ├── feature_extraction.py      # Feature extraction using ProtGen, ESM-2, ProtT5-BFD, ProtT5-XL
│   ├── GAC_BiT_CNN_Training.py    # Model training with 5-fold cross-validation
│   ├── GAC_BiT_CNN_Testing.py     # Independent test set evaluation
│   └── fuse_features.py           # Fuse multiple embedding CSVs into one feature set
│
├── results/
│   └── sample_output.txt          # Sample output metrics from 5-fold cross-validation
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── LICENSE                        # MIT License
```

---

## 🗂️ Dataset Description

All datasets are provided in **FASTA format**.

| File | Positives (PMT) | Negatives (non-PMT) | Total |
|------|----------------|---------------------|-------|
| `data/data_training.txt` | 2,290 | 2,290 | 4,580 |
| `data/data_testing.txt`  | 572   | 572   | 1,144 |
| `data/external_dataset.txt` | Variable | Variable | Variable |

**Dataset Construction:**
- Positive sequences retrieved from NCBI using query "Methyltransferase"
- Negative sequences retrieved from NCBI (non-methyltransferase proteins)
- Redundancy removed using **CD-HIT** at **40% sequence identity threshold**
- Sequences shorter than 50 amino acids were excluded

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Farman335/GAC-BiT-CNN-MT.git
cd GAC-BiT-CNN-MT
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Feature Extraction

Extract embeddings from protein sequences using all four protein language models:

```bash
python code/feature_extraction.py
```

This will generate four CSV files:
- `ProtGen_embeddings.csv` (1024 features)
- `ESM2_embeddings.csv` (1280 features)
- `ProtT5_BFD_embeddings.csv` (1024 features)
- `ProtT5_XL_UNIREF50_embeddings.csv` (1024 features)

### Step 2 — Fuse Feature Sets

Combine all four embeddings into one fused feature vector (4352-dim):

```bash
python code/fuse_features.py
```

### Step 3 — Train the Model

Train GAC-BiT-CNN using 5-fold cross-validation:

```bash
python code/GAC_BiT_CNN_Training.py
```

The trained model will be saved as:
- `GAC_BiTCN.keras` — Full model
- `GAC_BiTCN.weights.h5` — Weights only
- `GAC_BiTCN.json` — Model architecture

### Step 4 — Evaluate on Independent Test Set

```bash
python code/GAC_BiT_CNN_Testing.py
```

---

## 🧬 Model Architecture

```
Input (Fused Embeddings: 4352-dim)
         │
    ┌────┴─────────────────────────┐
    │                              │
  CNN Branch                  BiT Branch           GAC Branch
(Multi-scale TCN)         (Bidirectional       (GAN + Capsule
 kernel: 3,5,7            Transformer +         Network with
 dilation: 1,2,3          BiLSTM)               Dynamic Routing)
    │                              │                   │
    └──────────────────────────────┴───────────────────┘
                              │
                     Concatenate Features
                              │
                     Dense (128) + Dropout(0.3)
                              │
                     Sigmoid Output (Binary)
```

### Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Capsule Network | Primary Capsule Dim | 8 |
| Capsule Network | Num Capsules | 4 |
| Capsule Network | Routing Iterations | 3 |
| Capsule Network | Dropout | 0.2 |
| CNN Branch | Kernel Sizes | (3, 5, 7) |
| CNN Branch | Dilation Rates | (1, 2, 3) |
| CNN Branch | Filters | 64 |
| BiT Branch | Attention Heads | 2 |
| BiT Branch | Key Dimension | 32 |
| BiT Branch | LSTM Units | 64 |
| GAN | Generator LR | 0.0005 |
| GAN | Discriminator LR | 0.0001 |
| Training | Batch Size | 32 |
| Training | Learning Rate | 1e-4 |
| Training | Max Epochs | 100 (early stop) |

---

## 📦 Protein Language Models Used

| Model | Embedding Dim | Pretraining Database |
|-------|--------------|---------------------|
| ProtGen (progen2-medium) | 1024 | UniRef90 + BFD (280M sequences) |
| ESM-2 (esm2_t33_650M_UR50D) | 1280 | UniRef90 (250M sequences) |
| ProtT5-BFD (prot_t5_xl_bfd) | 1024 | BFD (2.1B sequences) |
| ProtT5-XL-UniRef50 (prot_t5_xl_uniref50) | 1024 | UniRef50 (curated) |
| **Fused (Concatenated)** | **4352** | All four databases |

---

## 📈 Comparison with State-of-the-Art

| Method | Accuracy | Sensitivity | Specificity | AUC | MCC |
|--------|----------|-------------|-------------|-----|-----|
| PMTPred [Yadav et al., 2024] | 87.94% | 89.38% | 88.48% | 90.40% | 0.759 |
| **GAC-BiT-CNN (Proposed)** | **98.16%** | **97.89%** | **98.42%** | **99.74%** | **0.963** |

---

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:

- Python >= 3.8
- TensorFlow >= 2.10
- PyTorch >= 1.13
- Transformers >= 4.30
- scikit-learn >= 1.0
- Biopython >= 1.79
- pandas, numpy, matplotlib

---

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{muazzam2025gacbitcnn,
  title={Enhancing Functional Prediction of Methyltransferases Using a Hybrid 
         GAC-BiT-CNN Deep Learning Model with Protein Language Model Embeddings},
  author={Muazzam, Muhammad and Ali, Farman and Asif, Muhammad},
  journal={},
  year={2025},
  publisher={}
}
```

---

## 📬 Contact

- **Muhammad Muazzam** — muazzammuhammad043@gmail.com
- **Farman Ali** — farman335@yahoo.com | farman.buic@bahria.edu.pk

Department of Computer Science, Bahria University Islamabad, Pakistan

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
