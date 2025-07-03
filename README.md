# Mutual Information Estimation Library

This Python library provides flexible tools to compute **mutual information (MI)**, **conditional mutual information (CMI)**, and **entropy (H)** using either:

- **Binned estimation** (quantile or regular-width)
- **Kernel Density Estimation (KDE)** for continuous variables

It supports 1D variables, conditional information, and pairwise MI matrix computation.

## Installation

This is a standalone library. To use, clone/download and place the `.py` files in your project.

Dependencies:

```bash
pip install numpy scikit-learn scipy
```

## Usage

### 1. Mutual Information

```python
from mutual_information import MI

mi = MI(x, y, method='kde', bins=100)  # or 'uniform_binned', 'regular_binned'
```

### 2. Conditional Mutual Information

```python
from mutual_information import CMI

cmi = CMI(x, y, z, method='uniform_binned', bins=10)
```

### 3. Entropy

```python
from mutual_information import H

entropy = H(x, method='kde', bins=100)
```

### 4. MI Matrix Between All Columns in a Dataset

```python
from mutual_information import MI_matrix

mi_mat = MI_matrix(X, method='regular_binned', bins=5)
```

## Methods

| Method           | Description                                      |
|------------------|--------------------------------------------------|
| `uniform_binned` | Equal-frequency binning (quantile-based)         |
| `regular_binned` | Equal-width binning                              |
| `kde`            | Kernel Density Estimation (continuous variables) |

## File Structure

```
binning.py               # Binning-based MI and entropy estimators
kde.py                   # KDE-based MI and entropy estimators
mutual_information.py    # Main API for MI, CMI, H, and MI matrix
```

## Example

```python
import numpy as np
from mutual_information import MI, CMI, H

x = np.random.normal(size=1000)
y = x + 0.1 * np.random.normal(size=1000)
z = np.random.normal(size=1000)

print("MI(x, y):", MI(x, y))
print("CMI(x, y | z):", CMI(x, y, z))
print("Entropy(x):", H(x))
```

## License

MIT License (you can adapt this if needed)

## Author

Developed by Nathan Burton — for information-theoretic research and experimentation.
