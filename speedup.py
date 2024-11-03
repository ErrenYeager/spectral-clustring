from spectral import spectral_clustering as spectral_clustering_old
from mnist import construct_affinity_matrix as construct_affinity_matrix_old

from metrics import clustering_score

from numba import jit, njit, prange, vectorize, guvectorize, cuda

import numpy as np
from numpy import linalg as LA
from timeit import timeit
from timeit import default_timer as timer


# @njit
# def compute_min_distances(p1, p2):
#     m, n = p1.shape[0], p2.shape[0]
#     min_distances = np.empty(m)
#     for i in range(m):
#         min_dist = np.inf
#         for j in range(n):
#             dist = 0
#             for k in range(p1.shape[1]):  # Assuming p1 and p2 have the same number of columns
#                 diff = p1[i, k] - p2[j, k]
#                 dist += diff * diff
#             if dist < min_dist:
#                 min_dist = dist
#         min_distances[i] = np.sqrt(min_dist)
#     return min_distances
#
#
# # TODO: Rewrite the chamfer_distance function
# @njit
# def chamfer_distance(point_cloud1, point_cloud2):
#     distances_1_to_2 = compute_min_distances(point_cloud1, point_cloud2)
#     distances_2_to_1 = compute_min_distances(point_cloud2, point_cloud1)
#     chamfer_dist = np.mean(distances_1_to_2) + np.mean(distances_2_to_1)
#     return chamfer_dist
#
#
# # TODO: Rewrite the spectral_clustering function
# @njit
# def spectral_clustering(affinity, k):
#     L_sym = laplacian(affinity)
#     eigvals, eigvecs = np.linalg.eigh(L_sym)
#     selected_eigvecs = eigvecs[:, :k]
#     labels, centroids = k_means_clustering(selected_eigvecs, k)
#     return labels
#
#
# # TODO: Rewrite the rigid_transform function
# @njit
# def rigid_transform(A, B):
#     assert A.shape == B.shape
#     m = A.shape[0]
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     AA = A - centroid_A
#     BB = B - centroid_B
#     H = np.dot(AA.T, BB)
#     U, S, Vt = LA.svd(H)
#     R = np.dot(Vt.T, U.T)
#     if LA.det(R) < 0:
#         Vt[m - 1, :] *= -1
#         R = np.dot(Vt.T, U.T)
#     t = centroid_B.T - np.dot(R, centroid_A.T)
#     return R, t
#
#
# #
# # # TODO: Rewrite the k_means_clustering function
# # @njit
# # def k_means_clustering(data, k, max_iterations=100):
# #     random_indices = np.random.choice(data.shape[0], k, replace=False)
# #     centroids = data[random_indices]
# #     for _ in range(max_iterations):
# #         centroids_expanded = centroids[np.newaxis, :, :]
# #         data_expanded = data[:, np.newaxis, :]
# #         differences = data_expanded - centroids_expanded
# #         squared_distances = np.sum(differences ** 2, axis=2)
# #         distances = np.sqrt(squared_distances)
# #         labels = np.argmin(distances, axis=1)
# #         new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
# #         if np.allclose(centroids, new_centroids):
# #             break
# #         centroids = new_centroids
# #         return labels, centroids
#
# def k_means_clustering(data, k, max_iterations=100):
#     # Randomly initialize centroids
#     random_indices = np.random.choice(data.shape[0], k, replace=False)
#     centroids = data[random_indices].astype(np.float64)
#
#     for _ in range(max_iterations):
#         # Compute distances between each data point and centroids
#         distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
#         # Assign each data point to the closest centroid
#         labels = np.argmin(distances, axis=0)
#         # Update centroids
#         new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
#         # Check for convergence
#         if np.allclose(centroids, new_centroids):
#             break
#         centroids = new_centroids
#
#     return labels, centroids
#
#
# # TODO: Rewrite the construct_affinity_matrix function
# @njit(parallel=True)
# def construct_affinity_matrix(point_clouds):
#     num_clouds = point_clouds.shape[0]
#     affinity_matrix = np.zeros((num_clouds, num_clouds))
#
#     for i in prange(num_clouds):
#         for j in prange(i + 1, num_clouds):
#             distance = chamfer_distance(point_clouds[i], point_clouds[j])
#             similarity = np.exp(-distance)
#             affinity_matrix[i, j] = affinity_matrix[j, i] = similarity
#
#     return affinity_matrix
#
#
# # TODO: Rewrite the icp function
# @njit
# def icp(source, target, max_iterations=100, tolerance=1e-5):
#     transformed_source = np.copy(source)
#     prev_distance = np.inf
#
#     for _ in range(max_iterations):
#         distances = compute_min_distances(transformed_source, target)
#         nearest_neighbor_indices = np.argmin(distances, axis=1)
#         closest_points = target[nearest_neighbor_indices]
#
#         R, t = rigid_transform(transformed_source, closest_points)
#         transformed_source = np.dot(R, transformed_source.T).T + t
#
#         chamfer_dist = chamfer_distance(transformed_source, target)
#         if np.abs(prev_distance - chamfer_dist) < tolerance:
#             break
#         prev_distance = chamfer_dist
#
#     return R, t, transformed_source
#
#
# # TODO: Rewrite the laplacian function
# @njit
# def laplacian(A):
#     n = A.shape[0]
#     D = np.zeros((n, n))
#     np.fill_diagonal(D, np.sum(A, axis=1))
#     D_sqrt_inv = np.linalg.inv(np.sqrt(D))
#     L_sym = np.eye(n) - np.dot(D_sqrt_inv, np.dot(A, D_sqrt_inv))
#     return L_sym

@njit
def k_means_clustering(data, k, max_iterations=100):
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_indices]

    for _ in range(max_iterations):
        centroids_expanded = centroids[np.newaxis, :, :]
        data_expanded = data[:, np.newaxis, :]
        differences = data_expanded - centroids_expanded
        squared_distances = np.sum(differences ** 2, axis=2)
        distances = np.sqrt(squared_distances)

        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            sum_points = np.zeros(data.shape[1])
            count = 0
            for j in range(data.shape[0]):
                if labels[j] == i:
                    sum_points += data[j]
                    count += 1
            if count != 0:
                new_centroids[i] = sum_points / count

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


@njit
def laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.linalg.inv(D_sqrt)
    D_sqrt_inv_dot_A = np.dot(D_sqrt_inv, A)
    D_sqrt_inv_dot_A_dot_D_sqrt_inv = np.dot(D_sqrt_inv_dot_A, D_sqrt_inv)
    L_sym = np.eye(A.shape[0]) - D_sqrt_inv_dot_A_dot_D_sqrt_inv
    return L_sym


@njit
def spectral_clustering(affinity, k):
    L_sym = laplacian(affinity)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    selected_eigvecs = eigvecs[:, :k]

    labels, centroids = k_means_clustering(selected_eigvecs, k)

    return labels


@njit
def compute_min_distances(p1, p2):
    min_distances = np.zeros(p1.shape[0])
    for i in range(p1.shape[0]):
        min_distance = np.inf
        for j in range(p2.shape[0]):
            distance = norm(p1, p2)
            if distance < min_distance:
                min_distance = distance
        min_distances[i] = min_distance
    return min_distances


@njit
def norm(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2)  # Calculate the squared differences
    dist = np.sqrt(squared_dist)  # Take the square root of the sum
    return dist


@njit
def chamfer_distance(point_cloud1, point_cloud2):
    distances_1_to_2 = compute_min_distances(point_cloud1, point_cloud2)
    distances_2_to_1 = compute_min_distances(point_cloud2, point_cloud1)
    chamfer_dist = np.mean(distances_1_to_2) + np.mean(distances_2_to_1)
    return chamfer_dist


@njit
def rigid_transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t


@njit
def icp(source, target, max_iterations=100, tolerance=1e-5):
    transformed_source = np.copy(source)
    prev_distance = np.inf

    for _ in range(max_iterations):
        distances = compute_min_distances(transformed_source, target)
        nearest_neighbor_indices = np.argmin(distances, axis=1)
        closest_points = target[nearest_neighbor_indices]

        R, t = rigid_transform(transformed_source, closest_points)

        transformed_source = np.dot(R, transformed_source.T).T + t

        chamfer_dist = chamfer_distance(transformed_source, target)

        if np.abs(prev_distance - chamfer_dist) < tolerance:
            break
        prev_distance = chamfer_dist

    return R, t, transformed_source


@njit
def construct_affinity_matrix(point_clouds):
    num_clouds = point_clouds.shape[0]
    affinity_matrix = np.zeros((num_clouds, num_clouds))

    for i in range(num_clouds):
        for j in range(i + 1, num_clouds):
            distance = chamfer_distance(point_clouds[i], point_clouds[j])
            similarity = np.exp(-distance)
            affinity_matrix[i, j] = affinity_matrix[j, i] = similarity

    return affinity_matrix


if __name__ == "__main__":
    dataset = np.load("datasets/mnist.npz")
    X = dataset['data']  # feature points
    y = dataset['target']  # ground truth labels
    n = len(np.unique(y))  # number of clusters

    # TODO: Run both the old and speed up version of your algorithms and capture running time
    # Measure old version
    # start_time_old = timer()
    # Ach_old = construct_affinity_matrix_old(X)
    # y_pred_old = spectral_clustering_old(Ach_old, n)
    # old_time = timer() - start_time_old
    # print("Old version time:", old_time)

    # Measure new version
    start_time_new = timer()
    print('hi')
    Ach_new = construct_affinity_matrix(X)
    print('hi 2')
    y_pred_new = spectral_clustering(Ach_new, n)
    new_time = timer() - start_time_new

    # TODO: Compare the running time using timeit module
    print("New version time:", new_time)

    # Optionally, compare clustering results
    # print("Old Chamfer affinity on MNIST:", clustering_score(y, y_pred_old))
    print("New Chamfer affinity on MNIST:", clustering_score(y, y_pred_new))
