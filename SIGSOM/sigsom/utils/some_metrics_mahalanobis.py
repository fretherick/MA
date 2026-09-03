import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from scipy.spatial.distance import mahalanobis
import functools
from sklearn.metrics.pairwise import pairwise_distances_chunked




def calinski_harabasz_score_mahalanobis(X, labels):
    """
    Compute the Calinski and Harabasz score using Mahalanobis distance.

    The score is defined as the ratio of the sum of between-cluster dispersion
    and of within-cluster dispersion using Mahalanobis distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    # Compute the covariance matrix and its inverse
    cov_matrix = np.cov(X, rowvar=False)+ np.eye(X.shape[1]) * 1e-6
    cov_matrix_inv = np.linalg.inv(cov_matrix) 

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)

    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        # Between-cluster dispersion using Mahalanobis distance
        extra_disp += len(cluster_k) * mahalanobis(mean_k, mean, cov_matrix_inv)# ** 2
        
        # Within-cluster dispersion
        cov_matrix_within = np.cov(cluster_k, rowvar=False)+ np.eye(X.shape[1]) * 1e-6
        cov_matrix_inv_within = np.linalg.inv(cov_matrix_within)
        # Within-cluster dispersion using Mahalanobis distance
        for x in cluster_k:
            intra_disp += mahalanobis(x, mean_k, cov_matrix_inv_within)# ** 2

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )

#davies_bouldin_score_mahalanobis
def davies_bouldin_score_mahalanobis(X, labels):
    """
    Compute the Davies-Bouldin score using Mahalanobis distance.

    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances using Mahalanobis distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, n_features = X.shape
    n_labels = len(le.classes_)

    # Ensure there are more than one cluster
    if n_labels <= 1:
        raise ValueError("Davies-Bouldin index requires more than one cluster.")

    # Compute the covariance matrix and its inverse
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    # Initialize arrays for intra-cluster distances and centroids
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, n_features), dtype=float)

    # Compute intra-cluster distances and centroids
    for k in range(n_labels):
        cluster_k = X[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        cov_matrix_within = np.cov(cluster_k, rowvar=False)
        cov_matrix_inv_within = np.linalg.inv(cov_matrix_within)
        intra_dists[k] = np.mean([mahalanobis(x, centroid, cov_matrix_inv_within) for x in cluster_k])

    # Compute pairwise Mahalanobis distances between centroids
    centroid_distances = np.array([
        [mahalanobis(centroids[i], centroids[j], cov_matrix_inv) if i != j else np.inf
         for j in range(n_labels)] for i in range(n_labels)
    ])

    # Compute the Davies-Bouldin score
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def silhouette_score_mahalanobis(
    X, labels, sample_size=None, random_state=None
):
    """Compute the mean Silhouette Coefficient of all samples using Mahalanobis distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature array.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    sample_size : int, default=None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_size is not None``.
        Pass an int for reproducible results across multiple function calls.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    """
    X, labels = check_X_y(X, labels)
    
    if sample_size is not None:
        random_state = np.random.default_rng(random_state)
        indices = random_state.choice(X.shape[0], sample_size, replace=False)
        X, labels = X[indices], labels[indices]
    
    # Compute the covariance matrix and its inverse for Mahalanobis distance
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    
    return np.mean(silhouette_samples_mahalanobis(X, labels, cov_matrix_inv))


def silhouette_samples_mahalanobis(X, labels, cov_matrix_inv):
    """Compute the Silhouette Coefficient for each sample using Mahalanobis distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature array.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    cov_matrix_inv : array-like of shape (n_features, n_features)
        Inverse of the covariance matrix of the dataset.

    Returns
    -------
    silhouette : array-like of shape (n_samples,)
        Silhouette Coefficients for each sample.
    """
    X, labels = check_X_y(X, labels, accept_sparse=["csr"])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)
    
    # Define a function for pairwise distances using Mahalanobis distance
    def mahalanobis_pairwise(X):
        n = X.shape[0]
        D = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = mahalanobis(X[i], X[j], cov_matrix_inv)
        return D

    # Compute the pairwise distances for silhouette calculation
    distances = mahalanobis_pairwise(X)
    reduce_func = functools.partial(
        _silhouette_reduce_mahalanobis, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(distances, reduce_func=reduce_func, metric='precomputed'))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def _silhouette_reduce_mahalanobis(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for a vertical chunk of X using Mahalanobis distance.

    Parameters
    ----------
    D_chunk : array-like of shape (n_chunk_samples, n_samples)
        Precomputed Mahalanobis distances for a chunk.

    start : int
        First index in the chunk.

    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.

    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    n_chunk_samples = D_chunk.shape[0]
    cluster_distances = np.zeros(
        (n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype
    )

    for i in range(n_chunk_samples):
        sample_weights = D_chunk[i]
        sample_labels = labels
        cluster_distances[i] += np.bincount(
            sample_labels, weights=sample_weights, minlength=len(label_freqs)
        )

    # intra_index selects intra-cluster distances within cluster_distances
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    # intra_cluster_distances are averaged over cluster size outside this function
    intra_cluster_distances = cluster_distances[intra_index]
    # of the remaining distances we normalize and extract the minimum
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances

def dunn_index_mahalanobis(X, labels):
    """
    Compute the Dunn Index using Mahalanobis distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of n_features-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    dunn_index : float
        The resulting Dunn Index. Higher values indicate better clustering.

    """
    # Check and prepare data
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)

    if n_labels < 2:
        raise ValueError("Dunn Index is not defined for less than 2 clusters.")

    # Compute the covariance matrix and its inverse
    cov_matrix = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    # Compute intra-cluster distances (maximum within each cluster)
    intra_dists = []
    for k in range(n_labels):
        cluster_k = X[labels == k]
        cov_matrix_within = np.cov(cluster_k, rowvar=False) + np.eye(cluster_k.shape[1]) * 1e-6
        cov_matrix_inv_within = np.linalg.inv(cov_matrix_within)
        if cluster_k.shape[0] > 1:
            dists = [
                mahalanobis(cluster_k[i], cluster_k[j], cov_matrix_inv_within)
                for i in range(len(cluster_k))
                for j in range(i + 1, len(cluster_k))
            ]
            intra_dists.append(max(dists) if dists else 0)
        else:
            intra_dists.append(0)
    max_intra_dist = max(intra_dists)

    # Compute inter-cluster distances (minimum between clusters)
    inter_dists = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            cluster_i = X[labels == i]
            cluster_j = X[labels == j]
            dists = [
                mahalanobis(x, y, cov_matrix_inv)
                for x in cluster_i
                for y in cluster_j
            ]
            inter_dists.append(min(dists) if dists else np.inf)
    min_inter_dist = min(inter_dists)

    # Compute Dunn Index
    dunn_index = min_inter_dist / max_intra_dist if max_intra_dist > 0 else 0
    return dunn_index
