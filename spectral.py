from kmeans import k_means_clustering
from numpy import linalg as LA
import numpy as np

import numpy as np


def laplacian(A):
    """
    Calculate the Laplacian matrix of the affinity matrix A using the symmetric normalized Laplacian formulation.

    Parameters:
    - A: numpy array, affinity matrix capturing pairwise relationships between data points.

    Returns:
    - L_sym: numpy array, symmetric normalized Laplacian matrix.
    """

    # TODO: Calculate degree matrix
    D = np.diag(np.sum(A, axis=1))

    # TODO: Calculate the inverse square root of the symmetric matrix
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.linalg.inv(D_sqrt)

    # TODO: Return symmetric normalized Laplacian matrix
    D_sqrt_inv_dot_A = np.dot(D_sqrt_inv, A)
    D_sqrt_inv_dot_A_dot_D_sqrt_inv = np.dot(D_sqrt_inv_dot_A, D_sqrt_inv)
    L_sym = np.eye(A.shape[0]) - D_sqrt_inv_dot_A_dot_D_sqrt_inv

    return L_sym


def spectral_clustering(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """

    # TODO: Compute Laplacian matrix
    laplacian_matrix = laplacian(affinity)

    # TODO: Compute the first k eigenvectors of the Laplacian matrix
    eigen_values, eigen_vectors = np.linalg.eigh(laplacian_matrix)
    chosen_eigen_vectors = eigen_vectors[:, :k]

    # TODO: Apply K-means clustering on the selected eigenvectors
    labels, centroids = k_means_clustering(chosen_eigen_vectors, k)

    # TODO: Return cluster labels
    return labels
