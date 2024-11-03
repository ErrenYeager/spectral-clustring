import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score


def compute_min_distances(p1, p2):
    # Compute all pairwise distances between two sets of points
    dist_matrix = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2))
    # Find the minimum distance for each point in p1 to points in p2
    min_distances = np.min(dist_matrix, axis=1)
    return min_distances


def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer distance between two point clouds.

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - dist: float, the Chamfer distance between the two point clouds.
    """

    # TODO: Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2
    distances_1_to_2 = compute_min_distances(point_cloud1, point_cloud2)

    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1
    distances_2_to_1 = compute_min_distances(point_cloud2, point_cloud1)

    # TODO: Return Chamfer distance, sum of the average distances in both directions
    chamfer_dist = np.mean(distances_1_to_2) + np.mean(distances_2_to_1)
    return chamfer_dist


def rigid_transform(A, B):
    """
    Find the rigid (translation + rotation) transformation between two sets of points.

    Parameters:
    - A: numpy array, mxn representing m points in an n-dimensional space.
    - B: numpy array, mxn representing m points in an n-dimensional space.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    """

    assert A.shape == B.shape

    # Number of points
    m = A.shape[0]

    # TODO: Subtract centroids to center the point clouds A and B
    centroids_of_A = np.mean(A, axis=0)
    centroids_of_B = np.mean(B, axis=0)
    AA = A - centroids_of_A
    BB = B - centroids_of_B

    # TODO: Construct Cross-Covariance matrix
    H = np.dot(AA.T, BB)

    # TODO: Apply SVD to the Cross-Covariance matrix
    U, Sigma, V_transpose = LA.svd(H)

    # TODO: Calculate the rotation matrix
    R = np.dot(V_transpose.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        V_transpose[m - 1, :] *= -1
        R = np.dot(V_transpose.T, U.T)

    # TODO: Calculate the translation vector
    t = centroids_of_B.T - np.dot(R, centroids_of_A.T)

    # TODO: Return rotation and translation matrices
    return R, t


def icp(source, target, max_iterations=100, tolerance=1e-5):
    """
        Perform ICP (Iterative Closest Point) between two sets of points.

        Parameters:
        - source: numpy array, mxn representing m source points in an n-dimensional space.
        - target: numpy array, mxn representing m target points in an n-dimensional space.
        - max_iterations: int, maximum number of iterations for ICP.
        - tolerance: float, convergence threshold for ICP.

        Returns:
        - R: numpy array, n x n rotation matrix.
        - t: numpy array, translation vector.
        - transformed_source: numpy array, mxn representing the transformed source points.
        """

    transformed_source = np.copy(source)
    prev_distance = np.inf

    # TODO: Iterate until convergence
    for _ in range(max_iterations):
        # TODO: Find the nearest neighbors of target in the source
        # Find nearest neighbors in the target for each point in the transformed source
        distances = compute_min_distances(transformed_source, target)
        nearest_neighbor_indices = np.argmin(distances, axis=1)
        closest_points = target[nearest_neighbor_indices]

        # TODO: Calculate rigid transformation
        # Calculate rigid transformation that best aligns the transformed source to its nearest neighbors in the target
        R, t = rigid_transform(transformed_source, closest_points)

        # TODO: Apply transformation to source points
        transformed_source = np.dot(R, transformed_source.T).T + t

        # TODO: Calculate Chamfer distance
        # Calculate Chamfer distance for convergence check
        chamfer_dist = chamfer_distance(transformed_source, target)

        # TODO: Check for convergence
        if np.abs(prev_distance - chamfer_dist) < tolerance:
            break  # Convergence achieved
        prev_distance = chamfer_dist

    # TODO: Return the transformed source
    return R, t, transformed_source


def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """

    num_clouds = point_clouds.shape[0]
    affinity_matrix = np.zeros((num_clouds, num_clouds))

    # TODO: Iterate over point clouds to fill affinity matrix
    for i in range(num_clouds):
        # TODO: For each pair of point clouds, register them with each other
        # TODO: Calculate symmetric Chamfer distance between registered clouds
        for j in range(i + 1, num_clouds):
            diss_similarity_score = chamfer_distance(point_clouds[i], point_clouds[j])
            similarity_score = np.exp(-diss_similarity_score)  # using the formula used in task 3.1
            affinity_matrix[i, j] = affinity_matrix[j, i] = similarity_score

    return affinity_matrix


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset["data"]
    y = dataset["target"]
    n = len(np.unique(y))

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Plot Ach using its first 3 eigenvectors
    # Eigen decomposition
    eigenvalues, eigenvectors = LA.eig(Ach)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx][:, :3]  # Select first 3 eigenvectors

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i, unique_label in enumerate(np.unique(y_pred)):
        ax.scatter(eigenvectors[y_pred == unique_label, 0],
                   eigenvectors[y_pred == unique_label, 1],
                   eigenvectors[y_pred == unique_label, 2],
                   c=colors[i], label=f'Cluster {unique_label}')
    ax.legend()
    plt.title('3D Visualization using First 3 Eigenvectors')
    plt.show()
