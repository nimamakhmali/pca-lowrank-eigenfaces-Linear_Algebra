# pca-lowrank-eigenfaces-Linear_Algebra-
 A linear algebra–driven implementation of PCA via SVD, focusing on low-rank modeling, spectral analysis, and image reconstruction.



# PCA Low-Rank Modeling & Eigenfaces

## Overview

This project explores Principal Component Analysis (PCA) from a linear algebra perspective, with a focus on:

- Singular Value Decomposition (SVD)
- Low-rank matrix approximation
- Spectral analysis
- Image compression and reconstruction
- Eigenfaces visualization

The goal is to demonstrate how fundamental linear algebra concepts such as:

- Vector spaces
- Orthogonality
- Rank
- Eigenvalues
- Matrix factorization

come together to form one of the most important tools in machine learning and data analysis.

---

## Project Structure

```text
src/            Core linear algebra implementations
experiments/    Reproducible numerical experiments
notebooks/      Interactive demonstration
results/        Generated figures and outputs
data/           Dataset handling utilities
```

---

## Dataset

We use the Olivetti Faces dataset:

```python
from sklearn.datasets import fetch_olivetti_faces
```

Each image is treated as a high-dimensional vector.  
The dataset is centered before applying SVD-based PCA.

---

## Mathematical Background

Given a centered data matrix:

$$
B \in \mathbb{R}^{m \times n}
$$

We compute its Singular Value Decomposition:

$$
B = U S V^T
$$

The principal components correspond to the columns of $V$.

Low-rank approximation is obtained via:

$$
B_k = U_k S_k V_k^T
$$

This provides the best rank-$k$ approximation in the Frobenius norm sense.

---

## Experiments

The repository includes:

- Spectral decay analysis
- Reconstruction error vs. rank
- Eigenfaces visualization
- Image compression experiments
- Denoising via low-rank approximation

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run main script:

```bash
python main.py
```

Or explore the interactive notebook:

```text
notebooks/demo.ipynb
```

---

## Educational Purpose

This repository is designed for instructional use in linear algebra and machine learning courses.  

It demonstrates how abstract mathematical concepts translate into practical computational tools.

---

## License

This project is released for educational purposes under the MIT License.
