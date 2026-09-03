import torch
import numpy as np

def som_loss(weights, distances):
    """
    SOM loss

    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    #print(f"the distances are: {distances.shape} and the weights are: {weights.shape}")
    return torch.mean(torch.sum(weights*distances, dim=1))


def kmeans_loss(y_pred, distances):
    """
    k-means reconstruction loss

    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
    # Return
        k-means reconstruction loss
    """
    return torch.mean(torch.tensor([distances[i, y_pred[i]] for i in range(len(y_pred))]))

def quantization_error(d):
    """
    Calculate k-means quantization error (internal DESOM function)
    """
    # return torch.mean(torch.min(d, dim=1)[0])# [0] to get values and not indices, not necessary to specify more for clarity as sometimes [1] is used
    return torch.sum((torch.min(d, dim=1)[0]))

def topographic_error(d, map_size):
    """
    Calculate SOM topographic error (internal DESOM function)
    Topographic error is the ratio of data points for which the two best matching units are not neighbors on the map.
    """
    h, w = map_size

    def is_adjacent(k, l):
        if w > 1:
            return (abs(k//w-l//w) == 1 and abs(k % w - l % w) == 0) or (abs(k//w-l//w) == 0 and abs(k % w - l % w) == 1)
        else:
            return (abs(k // h - l // h) == 1 and abs(k % h - l % h) == 0) or (abs(k // h - l // h) == 0 and abs(k % h - l % h) == 1)
        
    btmus = torch.argsort(d, dim=1)[:, :2]  # best two matching units

    return 1. - torch.mean(torch.tensor([is_adjacent(btmus[i, 0], btmus[i, 1]) for i in range(d.shape[0])],dtype=torch.float32))


from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

def combined_error(dist_fun, som, x=None, d=None):
    """Combined error.

    Parameters
    ----------
    dist_fun : function (k : int, l : int) => int
        distance function between units k and l on the map.
    som : array, shape = [n_units, dim]
        SOM code vectors.
    x : array, shape = [n_samples, dim]
        (optional) input samples.
    d : array, shape = [n_samples, n_units]
        (optional) euclidean distances between input samples and code vectors.

    Returns
    -------
    ce : float
        combined error  (lower is better)

    References
    ----------
    Kaski, S., & Lagus, K. (1996). Comparing Self-Organizing Maps.
    """
    if d is None:
        if x is None:
            raise ValueError('If distance matrix d is not given, x cannot be None!')
        else:
            d = euclidean_distances(x, som)
    # pairwise euclidean distances between neighboring SOM prototypes
    # distances between non-neighboring units are set to inf to force the path to follow neighboring units
    d_som = csr_matrix([[np.sqrt(np.sum(np.square(som[k] - som[l]))) if dist_fun(k, l) == 1 else np.inf
                        for l in range(som.shape[0])]
                        for k in range(som.shape[0])])
    tbmus = np.argsort(d, axis=1)[:, :2]  # two best matching units
    ces = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        ces[i] = d[i, tbmus[i, 0]]
        if dist_fun(tbmus[i, 0], tbmus[i, 1]) == 1:  # if BMUs are neighbors
            ces[i] += d_som[tbmus[i, 0], tbmus[i, 1]]
        else:
            ces[i] += shortest_path(csgraph=d_som,
                                    method='auto',
                                    directed=False,
                                    return_predecessors=False,
                                    indices=tbmus[i, 0])[tbmus[i, 1]]
    return np.mean(ces)

### Other metrics
