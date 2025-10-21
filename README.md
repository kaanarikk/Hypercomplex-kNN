# Hypercomplex kNN
**A Dimension-Adaptive k-Nearest Neighbors Architecture Using Elliptic Hypercomplex Distance Metrics**

---

## 📘 Overview
**Hypercomplex kNN** is a Python implementation of a dimension-adaptive *k-Nearest Neighbors* algorithm that operates in elliptic hypercomplex spaces.  
Unlike traditional kNN models that rely on Euclidean or Manhattan distances, this framework introduces an **elliptic hypercomplex metric** capable of capturing richer geometric relationships across complex, quaternion, and octonion domains.

This repository provides:
- Theoretical formulation of *n-dimensional elliptic hypercomplex numbers*  
- A custom **distance metric** that satisfies the metric axioms (positive-definite, symmetric, triangle-inequality)  
- An implementation of **Algorithm 1: Hypercomplex k-NN classification**  
- Benchmark comparisons against classical metrics (Euclidean, Manhattan, Minkowski, Cosine)

---

## Features
- ✅ **Elliptic Hypercomplex Metric**: Adaptive distance computation based on a tunable parameter *p < 0*  
- 🧭 **Dimension-Adaptive Framework**: Works in 2D (complex), 4D (quaternion), and 8D (octonion) domains  
- 🔍 **Metric Validation**: Proven metric properties for all n = 2ᵏ dimensions  
- 📊 **Benchmarking Suite**: Comparative evaluation with classical distance metrics  
- 🧠 **ML-Ready Design**: Compatible with scikit-learn style interfaces for easy experimentation  

---

## 📈 Mathematical Background
The algorithm defines a generalized elliptic hypercomplex number:

\[
Q = \sum_{i=0}^{n-1} a_i e_i,\quad a_i \in \mathbb{R},\; e_i^2 = 
\begin{cases}
p & \text{if } i \text{ is odd}\\
1 & \text{if } i \text{ is even}
\end{cases}
\]

The induced **hypercomplex metric** between two points \( Q_1, Q_2 \in E_n^p \) is:

\[
d(Q_1, Q_2) = 
\sqrt{
\sum_{i \text{ even}} (a_i - b_i)^2 - 
p \sum_{i \text{ odd}} (a_i - b_i)^2
}
\]

where \( p < 0 \) controls the curvature and scaling of odd-indexed components.

---

## 🧩 Algorithm Outline
```python
for each test sample Z:
    compute d(Z, Q_i) for all training samples Q_i in X
    select k samples with smallest distances
    assign class label by majority vote among neighbors
```

---

## 🧪 Experimental Setup
All experiments were conducted on **Google Colab** using:
- Python 3.13  
- scikit-learn 1.8.1  
- NumPy 2.2.0  
- Pandas 3.0.2  
- NVIDIA T4 GPU (16 GB VRAM)

Performance was evaluated using **accuracy**, **precision**, **recall**, and **F1-score**, along with confusion matrices for detailed analysis.
---

## 🧠 Authors
- **Kaan Arık** – Sakarya University of Applied Sciences  
- **Arzu Sürekçi** – Sakarya University of Applied Sciences  
- **Hidayet Hüda Kösal** – Sakarya University, Department of Mathematics  

---

## 📝 License
This project is released under the **MIT License**. You are free to use, modify, and distribute it with attribution.
