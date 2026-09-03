import torch
import signatory
import numpy as np


def extract_hmm_features(X: np.ndarray, l: int) -> np.ndarray:
    """
    Extract the 9 features from each row of X according to Algorithm 2.
    It returns a 3D array of shape (n_series, T, 9) for each time point in each series.
    """
    nb_paths, T = X.shape
    #  0: observation               (y_t)
    #  1: left absolute change      (|y_t - y_{t-1}|)
    #  2: right absolute change     (|y_{t+1} - y_t|)
    #  3: centered local mean       (mean of y in [t - l//2, t + l//2])
    #  4: centered local std        (std  of y in [t - l//2, t + l//2])
    #  5: left local mean           (mean of y in [t - l, t - 1])
    #  6: left local std            (std  of y in [t - l, t - 1])
    #  7: right local mean          (mean of y in [t + 1, t + l])
    #  8: right local std           (std  of y in [t + 1, t + l])
    features = np.full((nb_paths, T, 9), np.nan, dtype=float)
    for i in range(nb_paths):
        y = X[i]
        for t in range(T):
            # 0) observation
            features[i, t, 0] = y[t]
            # 1) left absolute change
            if t > 0: features[i, t, 1] = abs(y[t] - y[t - 1])
            # 2) right absolute change
            if t < T - 1: features[i, t, 2] = abs(y[t + 1] - y[t])
            # 3) centered local mean and 4) centered local std
            left_c = max(0, t - l // 2)
            right_c = min(T, t + l // 2 + 1)  # +1 since slicing is exclusive of end
            segment = y[left_c:right_c]
            if len(segment) > 0:
                features[i, t, 3] = np.mean(segment)
                features[i, t, 4] = np.std(segment, ddof=1)
            # 5) left local mean and 6) left local std
            if t >= l:
                segment_left = y[t - l: t]
                if len(segment_left) > 0:
                    features[i, t, 5] = np.mean(segment_left)
                    features[i, t, 6] = np.std(segment_left, ddof=1)
            # 7) right local mean and 8) right local std
            if t + l < T:
                segment_right = y[t + 1: t + 1 + l]
                if len(segment_right) > 0:
                    features[i, t, 7] = np.mean(segment_right)
                    features[i, t, 8] = np.std(segment_right, ddof=1)
    return features


def extract_hmm_features_vectorized(X: np.ndarray, l: int) -> np.ndarray:
    """
    Extract the 9 features for each time series (each row in X) using vectorized operations.
    For indices where the window would extend beyond the series boundaries,
    the corresponding feature is set to NaN.
    Output: 3D array of shape (n_series, T, 9) containing the features.
    """
    nb_paths, T = X.shape
    features = np.full((nb_paths, T, 9), np.nan, dtype=float)

    # 0. Observation
    features[:, :, 0] = X

    # 1. Left absolute change and 2. Right absolute change
    diff = np.diff(X, axis=1)
    left_change = np.concatenate([np.full((nb_paths, 1), np.nan), np.abs(diff)], axis=1)
    right_change = np.concatenate([np.abs(diff), np.full((nb_paths, 1), np.nan)], axis=1)
    features[:, :, 1] = left_change
    features[:, :, 2] = right_change

    # Precompute cumulative sums (and cumulative sums of squares) for fast window sum/std.
    # These arrays have shape (n_series, T+1).
    cumsum = np.hstack([np.zeros((nb_paths, 1)), np.cumsum(X, axis=1)])
    cumsum2 = np.hstack([np.zeros((nb_paths, 1)), np.cumsum(X**2, axis=1)])

    # 3 & 4. Centered local mean and std.
    half = l // 2
    t_idx = np.arange(T)
    # For each time t, define the window:
    start = np.maximum(0, t_idx - half)
    end = np.minimum(T, t_idx + half + 1)  # +1 since end is exclusive
    counts = end - start  # number of elements in each window; shape (T,)

    # Use np.take_along_axis to pick the right cumulative sum values for each t.
    sum_centered = np.take_along_axis(cumsum, end[None, :], axis=1) - np.take_along_axis(cumsum, start[None, :], axis=1)
    centered_mean = sum_centered / counts
    features[:, :, 3] = centered_mean

    # For the std we use the formula:
    #   var = (sum(x^2) - (sum(x)^2)/count) / (count - 1)
    sum2_centered = np.take_along_axis(cumsum2, end[None, :], axis=1) - np.take_along_axis(cumsum2, start[None, :], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        centered_var = (sum2_centered - (sum_centered ** 2) / counts) / (counts - 1)
    centered_std = np.sqrt(centered_var)
    # If the window has only one element, set std to NaN.
    mask = counts == 1
    if np.any(mask):
        centered_std[:, mask] = np.nan
    features[:, :, 4] = centered_std

    # 5 & 6. Left local mean and std for window [t-l, t)
    if T > l:
        t_valid = np.arange(l, T)
        sum_left = cumsum[:, t_valid] - cumsum[:, t_valid - l]
        left_mean = sum_left / l
        features[:, t_valid, 5] = left_mean

        sum2_left = cumsum2[:, t_valid] - cumsum2[:, t_valid - l]
        # When l==1, set std to nan (cannot compute sample std).
        if l > 1:
            left_var = (sum2_left - (sum_left ** 2) / l) / (l - 1)
            left_std = np.sqrt(left_var)
        else:
            left_std = np.full_like(left_mean, np.nan)
        features[:, t_valid, 6] = left_std

    # 7 & 8. Right local mean and std for window [t+1, t+l+1]
    if T > l:
        t_valid = np.arange(0, T - l)
        sum_right = cumsum[:, t_valid + l + 1] - cumsum[:, t_valid + 1]
        right_mean = sum_right / l
        features[:, t_valid, 7] = right_mean

        sum2_right = cumsum2[:, t_valid + l + 1] - cumsum2[:, t_valid + 1]
        if l > 1:
            right_var = (sum2_right - (sum_right ** 2) / l) / (l - 1)
            right_std = np.sqrt(right_var)
        else:
            right_std = np.full_like(right_mean, np.nan)
        features[:, t_valid, 8] = right_std
    return features




def extract_signature_features_centered(X: np.ndarray, l: int, depth: int = 2) -> np.ndarray:
    """
    Extract signature features over centered window [t-l, t+l].
    Input shape: (n_paths, T)
    Output shape: (n_paths, T, D_sig)
    """
    n_paths, T = X.shape
    X_torch = torch.from_numpy(X).float().unsqueeze(-1)  # (N, T, 1)
    sig_dim = signatory.signature_channels(channels=1, depth=depth)

    sig_feats = torch.full((n_paths, T, sig_dim), float('nan'))

    for t in range(l, T - l):
        segment = X_torch[:, t - l : t + l + 1, :]  # centered window of size 2l+1
        print(segment.shape)
        sig_window = signatory.signature(segment, depth=depth)
        print(sig_window)
        sig_feats[:, t, :] = sig_window

    return sig_feats.numpy()





def test_hmm_features() -> None:
    X = np.random.normal(0, 1, (5, 15))
    l = 4
    out1 = extract_hmm_features(X, l)
    out2 = extract_hmm_features_vectorized(X, l)
    if not np.allclose(out1, out2, equal_nan=True):
        raise ValueError("Outputs are different")
    print("The 2 outputs are identical")
    # NOTE: for out2[i] there are NaN's at the beginning and end of the series. The number of
    # elements that contain NaN is equal to the length of the window l (at both sides).
    return

