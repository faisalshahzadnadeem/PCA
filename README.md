# PCA
Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction, feature extraction, and data visualization. It transforms data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. Here’s a concise overview of PCA:

### Key Concepts:

1. **Dimensionality Reduction**:
   - PCA reduces the number of dimensions (features) in the data while retaining as much variability (information) as possible.
   - It is particularly useful when dealing with high-dimensional data where visualization and interpretation become challenging.

2. **Principal Components**:
   - Principal components are the directions of maximum variance in the data.
   - The first principal component captures the largest possible variance, and each succeeding component captures the highest remaining variance, orthogonal to the previous ones.

3. **Covariance Matrix**:
   - PCA involves computing the covariance matrix of the data to understand how variables vary with respect to each other.
   - The covariance matrix helps in identifying the eigenvectors (principal components) and eigenvalues (variances along the principal components).

4. **Eigenvalues and Eigenvectors**:
   - Eigenvalues indicate the amount of variance captured by each principal component.
   - Eigenvectors indicate the direction of the principal components in the feature space.

### Steps in PCA:

1. **Standardize the Data**:
   - Mean-center the data and scale to unit variance to ensure that each feature contributes equally to the analysis.

2. **Compute the Covariance Matrix**:
   - Calculate the covariance matrix to understand the relationships between different features.

3. **Eigen Decomposition**:
   - Perform eigen decomposition on the covariance matrix to obtain eigenvalues and eigenvectors.

4. **Sort Eigenvalues and Select Principal Components**:
   - Sort the eigenvalues in descending order and select the top k eigenvalues and their corresponding eigenvectors to form the principal components.

5. **Transform the Data**:
   - Project the original data onto the new k-dimensional subspace using the selected principal components.

### Applications of PCA:

- **Data Visualization**: PCA can reduce data to 2 or 3 dimensions for easy visualization.
- **Noise Reduction**: By keeping only the principal components with the highest variance, PCA can help in removing noise from data.
- **Feature Extraction**: PCA can be used to derive new features that are linear combinations of the original features, which may be more informative.
- **Preprocessing for Machine Learning**: Reducing dimensionality can help in improving the performance and reducing the computational cost of machine learning algorithms.

### Example in Python:

Here's a simple example using Python with scikit-learn:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratios:", pca.explained_variance_ratio_)
print("Principal Components:\n", X_pca)
```

In this example, the data is standardized, and PCA is applied to transform it into two principal components. The explained variance ratios provide information on the proportion of variance captured by each component.
