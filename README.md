# NanoCon

# Project Overview

5-Methylcytosine (5mc), a fundamental element of DNA methylation in eukaryotes, plays a vital role in gene expression regulation, embryonic development, and other biological processes. Although several computational methods have been proposed for detecting the base modifications in DNA like 5mc sites from Nanopore sequencing data, they face challenges including sensitivity to noise, and ignoring the imbalanced distribution of methylation sites in real-world scenarios. Here, we develop NanoCon, a deep hybrid network coupled with contrastive learning strategy to detect 5mc methylation sites from Nanopore reads. In particular, we adopted a contrastive learning module to alleviate the issues caused by imbalanced data distribution in nanopore sequencing, offering a more accurate and robust detection of 5mc sites. Evaluation results demonstrate that NanoCon outperforms existing methods, highlighting its potential as a valuable tool in genomic sequencing and methylation prediction. Additionally, we also verified the effectiveness of our representation learning ability on two datasets by visualizing the dimension reduction of the features of methylation and non-methylation sites from our NanoCon. Furthermore, cross-species and cross-5mc methylation motifs experiments indicated the robustness and the ability to perform transfer learning of our model. We hope this work can contribute to the community by providing a powerful and reliable solution for 5mc site detection in genomic studies.

# Quick Start

## Installation

- Clone the project to your local machine.
```bash
git clone https://github.com/your-username/NanoCon.git
```

- Environment
```bash
conda env create -f environment.yml
```

- Usage Example
```bash
python -m -1 train_slice.py
```
