import numpy as np
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score


def gaussian_kernel(distance, sigma=1.0):
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


def euclidean_distance(data):
    return np.sum(
        (data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=-1
    )


def construct_affinity_matrix(data, affinity_type="rbf", k=3, sigma=1.0):
    n_samples = data.shape[0]
    sq_dists = euclidean_distance(data)
    if affinity_type == "knn":
        # Calculate the Euclidean distance matrix
        distances = np.sqrt(sq_dists)
        # For each row, sort the distances ascendingly and get their indices
        sorted_indices = np.argsort(distances, axis=1)
        # Initialize the affinity matrix with zeros
        affinity_matrix = np.zeros((n_samples, n_samples))
        # Use indices to set entries to 1 for the k-nearest neighbors
        for i in range(n_samples):
            for j in sorted_indices[
                     i, 1: k + 1
                     ]:  # Skip the first index since it's the point itself
                affinity_matrix[i, j] = 1
                affinity_matrix[j, i] = 1  # Ensure symmetry

    elif affinity_type == "rbf":
        # Apply the Gaussian kernel
        affinity_matrix = gaussian_kernel(np.sqrt(sq_dists), sigma)

    else:
        raise ValueError("Invalid affinity matrix type. Choose either 'rbf' or 'knn'.")

    return affinity_matrix


def plot_clusters(X, y, title, ax):
    # Assuming y contains integer labels for clusters
    unique_labels = np.unique(y)
    for label in unique_labels:
        ax.scatter(X[y == label, 0], X[y == label, 1], label=str(label))
    ax.set_title(title)


if __name__ == "__main__":
    datasets = ["blobs", "circles", "moons"]

    # TODO: Create and configure plot
    fig, axs = plt.subplots(
        nrows=3, ncols=4, figsize=(20, 15)
    )  # Adjust figsize as needed
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing as needed

    index = 0
    for ds_name in datasets:
        dataset = np.load("datasets/%s.npz" % ds_name)  # Adjust path as necessary
        X = dataset["data"]  # feature points
        y = dataset["target"]  # ground truth labels
        n = len(np.unique(y))  # number of clusters

        # Assume you have defined the clustering functions earlier
        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, "rbf", sigma=1.0)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, "knn", k=3)
        y_knn = spectral_clustering(Aknn, n)

        print("--------------------------------------------------------------")
        print("GREEDY: K-means on %s:" % ds_name, clustering_score(y, y_km, "greedy"))
        print(
            "GREEDY: RBF affinity on %s:" % ds_name,
            clustering_score(y, y_rbf, "greedy"),
        )
        print(
            "GREEDY: KNN affinity on %s:" % ds_name,
            clustering_score(y, y_knn, "greedy"),
        )
        print("--------------------------------------------------------------")
        print("NMI: K-means on %s:" % ds_name, clustering_score(y, y_km, "nmi"))
        print("NMI: RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf, "nmi"))
        print("NMI: KNN affinity on %s:" % ds_name, clustering_score(y, y_knn, "nmi"))
        print("--------------------------------------------------------------")

        # TODO: Create subplots

        # Plotting ground truth
        plot_clusters(X, y, "Ground Truth: %s" % ds_name, axs[index, 0])

        # Plotting k-means clustering results
        plot_clusters(X, y_km, "K-means: %s" % ds_name, axs[index, 1])

        # Plotting RBF affinity results
        plot_clusters(X, y_rbf, "RBF affinity: %s" % ds_name, axs[index, 2])

        # Plotting KNN affinity results
        plot_clusters(X, y_knn, "KNN affinity: %s" % ds_name, axs[index, 3])
        index = index + 1

    # TODO: Show subplots
    plt.show()
