# MSPMLTC
Official implementation of MSPMLTC: A novel Multi-Source Partial Multi-Label Learning framework. It leverages tensor three-mode projection and cross-source label consistency to effectively uncover latent structures and handle noisy labels across heterogeneous sources.

---

## 💡 Usage

We provide a small synthetic multi-source partial multi-label dataset in `datasets/` for quick testing.
> For real-world experiments, please download full datasets from their official sources (see below).

> After downloading, place your dataset in `datasets/`. Datasets containing noisy labels are located in the `datasets_noise_labels/` folder.

> Run `main.py`, which contains the full implementation of the MSPMLTC model, its core algorithm, and evaluation code.

---

## 📖 Model Overview

MSPMLTC addresses the Multi-Source Partial Multi-Label (MSPML) learning problem, where instances are described by features from multiple heterogeneous sources and each instance is associated with a set of candidate labels (only some of which are correct). The framework consists of three key components:

- **Tensor Three-Mode Projection**: Learns representation matrices from each source's feature space, stacks them as frontal slices of a representation tensor, and applies three-mode projections (instance-mode, label-mode, source-mode) to reveal relationships among instances, labels, and sources.

- **TRPCA-based Label Decomposition**: Decomposes observed labels into low-rank ground-truth labels and sparse noisy labels using Tensor Robust Principal Component Analysis (TRPCA), effectively separating clean signals from noise.

- **Cross-Source Label Consistency**: Enforces local topological structure preservation in the fused representation space, ensuring that semantically similar instances receive consistent labels across sources.

The model is optimized via an efficient Alternating Direction Method of Multipliers (ADMM) algorithm with guaranteed convergence.

---

## 🔗 Datasets Used in the Paper

> **Please download the datasets from their official sources listed below.**

All datasets used in our experiments are publicly available. Please download them directly from the original providers:

| Dataset Name | Description | Official Download Link |
|---|---|---|
| **Human** | Protein subcellular location (Human species) | https://www.uco.es/kdis/mllresources/ |
| **Rugby** | Event labeling in rugby match transcripts | https://github.com/transientlunatic/rugby-data |
| **Yeast** | Yeast protein functional classification | http://mulan.sourceforge.net/datasets-mlc.html |
| **Plant** | Protein subcellular location (Plant species) | https://www.uco.es/kdis/mllresources/ |
| **Pascal07** | Image classification (Pascal VOC 2007) | https://pjreddie.com/projects/pascal-voc-dataset-mirror/ |
| **3Sources** | Multi-view news articles (BBC, Reuters, Guardian) | http://mlg.ucd.ie/aggregation/index.html |
| **ESC-50** | Environmental sound classification (50 classes) | https://github.com/karolpiczak/ESC-50/tree/master?tab=readme-ov-file |
| **OBJECT** | Web image dataset (NUS-WIDE subset) | https://hyper.ai/datasets/16124 |
| **NOIZEUS** | Speech corpus with noise-augmented utterances | https://ecs.utdallas.edu/loizou/speech/noizeus/ |
| **SCENE** | Scene recognition from web images | http://mlg.ucd.ie/aggregation/index.html |

> 💡 After downloading, organize the data as needed and specify the path via command-line arguments or config files (see Usage above).

---

## 📊 Dataset Summary

The following table summarizes the statistics of the 10 datasets used in our experiments:

| Dataset | #Instances | #Sources | #Labels | Domain |
|---|---|---|---|---|
| Human | 3,106 | 3 | 14 | Biology |
| Rugby | 854 | 3 | 15 | Text |
| Yeast | 2,417 | 2 | 14 | Biology |
| Plant | 978 | 3 | 12 | Biology |
| Pascal07 | 9,963 | 6 | 20 | Image |
| 3Sources | 169 | 3 | 6 | Text |
| ESC-50 | 2,000 | 4 | 8 | Audio |
| OBJECT | 6,074 | 5 | 31 | Image |
| NOIZEUS | 150 | 4 | 7 | Audio |
| SCENE | 4,400 | 5 | 33 | Image |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/EalynChan/MSPMLTC.git
cd MSPMLTC
```
