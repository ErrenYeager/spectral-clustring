import numpy as np


def clustering_score(
    true_labels,
    predicted_labels,
    method='greedy'
):
    """
    Calculate the clustering score to assess the accuracy of predicted labels compared to true labels.

    Parameters:
    - true_labels: List or numpy array, true cluster labels for each data point.
    - predicted_labels: List or numpy array, predicted cluster labels for each data point.

    Returns:
    - score: float, clustering score indicating the accuracy of predicted labels.
    """

    # TODO: Calculate and return clustering score

    if method == "nmi":
        return normalized_mutual_information(true_labels, predicted_labels)
    elif method == "greedy":
        return greedy_clustering_score(true_labels, predicted_labels)
    else:
        raise ValueError("Method must be either 'nmi' or 'greedy'.")


def greedy_clustering_score(true_labels, predicted_labels):
    unique_true_labels = np.unique(true_labels)
    unique_predicted_labels = np.unique(predicted_labels)

    match_score = 0

    for utl in unique_true_labels:
        max_overlap = 0
        for upl in unique_predicted_labels:
            overlap = np.sum((true_labels == utl) & (predicted_labels == upl))
            max_overlap = max(max_overlap, overlap)
        match_score += max_overlap

    total_points = len(true_labels)
    score = match_score / total_points
    return score


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log(probabilities))


def mutual_information(true_labels, predicted_labels):
    mi = 0
    for t_label in np.unique(true_labels):
        for p_label in np.unique(predicted_labels):
            joint = np.sum(
                (true_labels == t_label) & (predicted_labels == p_label)
            ) / len(true_labels)
            if joint > 0:
                t_prob = np.sum(true_labels == t_label) / len(true_labels)
                p_prob = np.sum(predicted_labels == p_label) / len(predicted_labels)
                mi += joint * np.log(joint / (t_prob * p_prob))
    return mi


def normalized_mutual_information(true_labels, predicted_labels):
    mi = mutual_information(true_labels, predicted_labels)
    h_true = entropy(true_labels)
    h_pred = entropy(predicted_labels)
    return mi / np.sqrt(h_true * h_pred)
