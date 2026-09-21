# PCA from Scratch

> Principal Component Analysis implemented with NumPy, then extended to Kernel PCA with the RBF trick — and a demonstration of exactly where linear PCA fails.

PCA is the standard tool for linear dimensionality reduction. Kernel PCA extends
it to data whose structure is curved rather than flat. This notebook builds
both from the covariance matrix up, and uses `make_moons` — two interleaving
crescents — to show the difference concretely.

## Run it

Open [`Code PCA colab.ipynb`](Code%20PCA%20colab.ipynb) in Jupyter, or
[in Colab](https://colab.research.google.com/github/Fezzaioussama/PCA-from-scratch-using-Python/blob/main/TP01_machine_learning.ipynb).

```bash
pip install numpy matplotlib scikit-learn
```

The `code PCA` file holds the same implementation as a plain script.

## What it covers

| Section | Content |
|---|---|
| Introduction | Why linear vs. kernel methods diverge, and when each applies |
| PCA via covariance | Centre the data, build the covariance matrix, eigendecompose, project onto the top-`k` eigenvectors |
| Interpretation | Reading the eigenvalues as explained variance, and choosing `k` |
| Kernel PCA | The RBF kernel, the centred Gram matrix, and eigendecomposition in feature space |
| Comparison | Both applied to `make_moons`, where the classes are not linearly separable |

## The idea

**PCA** finds the directions along which the data varies most. Concretely:
centre the data, compute the covariance matrix, take its eigenvectors. The
eigenvector with the largest eigenvalue is the direction of greatest variance —
the first principal component — and projecting onto the top `k` of them keeps as
much variance as `k` dimensions can hold.

The catch is that this is a **rotation**. PCA can only find structure that a
linear projection can express. Give it two interleaving crescents and no
rotation will separate them, because the structure is curved.

**Kernel PCA** fixes this by running PCA in a higher-dimensional feature space
where the crescents *do* become separable — without ever computing the mapping.
The RBF kernel

```
k(xᵢ, xⱼ) = exp(−γ‖xᵢ − xⱼ‖²)
```

gives inner products in that space directly, so eigendecomposing the centred
kernel matrix yields the components. This is the kernel trick: the feature space
is implicit, potentially infinite-dimensional, and never materialised.

The cost is that you now trade one free parameter (`k`) for two (`k` and `γ`),
the components lose their interpretation as directions in the original space,
and the kernel matrix is n×n — so it scales with the number of samples, not the
number of features.

## Takeaway

Use PCA when the structure is linear: it's fast, interpretable, and scales with
dimensionality. Reach for Kernel PCA when a scatter plot shows curved structure
that no rotation will straighten — and expect to tune `γ`.
