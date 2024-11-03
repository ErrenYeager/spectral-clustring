import numpy as np


def k_means_clustering(data, k, max_iterations=100):
    # TODO: Randomly initialize centroids
    random_points = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_points]

    # TODO: Iterate until convergence and update centroids and labels
    for _ in range(max_iterations):
        # Compute distances between each data point and centroids
        centroids_expanded = centroids[np.newaxis, :, :]
        data_expanded = data[:, np.newaxis, :]
        differences = data_expanded - centroids_expanded
        squared_distances = np.sum(differences ** 2, axis=2)
        distances = np.sqrt(squared_distances)

        # Assign each data point to the closest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]  # Select points belonging to cluster i
            cluster_mean = cluster_points.mean(axis=0)  # Calculate mean of cluster points
            new_centroids.append(cluster_mean)  # Append mean to new centroids list

        new_centroids = np.array(new_centroids)  # Convert list of centroids to NumPy array

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # TODO: Return labels and centroids
    return labels, centroids
