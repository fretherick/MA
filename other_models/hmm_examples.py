

# Generate a sequence of observations from a hidden Markov model (HMM)
# Train a Jump Model on it

"""
Following the paper Nystrup, P., Lindström, E., & Madsen, H. (2020). Learning hidden Markov models with persistent
states by penalizing jumps, we simulate a Hidden Markov Model (HMM) and test a Jump Model over it.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from jumpmodels.jumpmodels.jump import JumpModel
from other_models.hmm_model import HiddenMarkovModel
from other_models.hmm_utils import extract_hmm_features_vectorized,extract_signature_features_centered
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import sys
print(print(sys.executable))
def paper_hmm(len_path: int = 250, nb_paths: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    # Define model with parameters from paper
    k = 2
    initial_state_prob = np.array([.85, .15])  # is this correct?
    transition_matrix = np.array([
        [0.9629, 0.0371],
        [0.2101, 0.7899],
    ])
    means = np.array([0.0123, -0.0157])
    variances = np.array([0.00120409, 0.00605284])

    # Simulate `nb_paths` sequences of length `len_path`
    hmm = HiddenMarkovModel(k, initial_state_prob, transition_matrix, means, variances)
    # np_states = np.empty((nb_paths, len_path))
    # np_log_rets = np.empty((nb_paths, len_path))
    # for i in range(nb_paths):
    #     np_states[i], np_log_rets[i] = hmm.simulate(T=len_path)
    np_states, np_log_rets = hmm.simulate_paths(nb_paths=nb_paths, T=len_path, dt=1, more_states=True)
    return np_states, np_log_rets

# from hmmlearn.hmm import GaussianHMM

# def paper_hmm(len_path: int = 250, nb_paths: int = 1000) -> tuple[np.ndarray, np.ndarray]:
#     # Hardy (2001) monthly regime-switching parameters
#     k = 2
#     means = np.array([[0.0123], [-0.0157]])
#     std_devs = np.array([0.0347, 0.0778])
#     variances = std_devs ** 2

#     transition_matrix = np.array([
#         [0.9629, 0.0371],
#         [0.2101, 0.7899]
#     ])

#     # Approximate stationary distribution
#     initial_state_prob = np.array([0.85, 0.15])

#     # Setup HMM
#     model = GaussianHMM(n_components=k, covariance_type="diag", init_params="")
#     model.startprob_ = initial_state_prob
#     model.transmat_ = transition_matrix
#     model.means_ = means
#     model.covars_ = variances.reshape(-1, 1)

#     # Simulate paths
#     np_states = np.empty((nb_paths, len_path), dtype=int)
#     np_log_rets = np.empty((nb_paths, len_path), dtype=float)

#     for i in range(nb_paths):
#         X, states = model.sample(len_path)
#         np_log_rets[i] = X[:, 0]
#         np_states[i] = states

#     return np_states, np_log_rets

if __name__ == "__main__":
    nb_paths = 1000
    len_path = 250
    windows = [5, 13]
    max_window = max(windows)
    nb_states = 2
    eff_timesteps = len_path-2*max_window

    # Output features
    OUT_FEAT = 9  # depends on the defined functions

    # Output containers
    feat_centers = np.full((nb_paths, nb_states, len(windows)*OUT_FEAT-3), fill_value=np.nan)
    labels = np.full((nb_paths, eff_timesteps), fill_value=np.nan)
    mean_per_cluster = np.full((nb_paths, nb_states), fill_value=np.nan)
    std_per_cluster = np.full((nb_paths, nb_states), fill_value=np.nan)


    np_states, np_log_rets = paper_hmm(len_path=len_path, nb_paths=nb_paths)
    # print(np_states)
    # print(np_log_rets)

    # Aggregate features coming from different windows
    all_features = np.empty((nb_paths, eff_timesteps, len(windows)*OUT_FEAT))

    for i, window in enumerate(windows):
        features = extract_signature_features_centered(X=np_log_rets, l=window)
        all_features = features
        print(features.shape)
        print(features)
        # all_features[:, :, i*OUT_FEAT:(i+1)*OUT_FEAT] = features[:, max_window:len_path - max_window]  # features without NaN's
    np_log_rets = np_log_rets[:, max_window:len_path-max_window]
    from sklearn.preprocessing import StandardScaler
    # 🔧 Standardize features per path
    for i in range(nb_paths):
        all_features[i] = StandardScaler().fit_transform(all_features[i])
    
    # indices_to_keep = [i for i in range(18) if i not in [9, 10, 11]]
    # all_features = all_features[:, :, indices_to_keep]

    # print(feat_centers.shape)

    #   display entire array
    # Define the Jump Model
    jump_penalty = 1
    # LOOP over the rows of np_states and np_log_rets
    balanced_accuracies = []
    for i in range(nb_paths):
        print(all_features[i].shape)
        jm = JumpModel(n_components=nb_states, jump_penalty=jump_penalty, cont=False)
        jm.fit(all_features[i], np_log_rets[i], sort_by="cumret")
        feat_centers[i] = jm.centers_
        labels[i] = jm.labels_
        mean_per_cluster[i] = jm.ret_
        std_per_cluster[i] = jm.vol_
        # there is also jm.val_ which includes the best value for the objective function
        print(jm.labels_)
        print(np_states[i])
    # Compute balanced accuracy for each path
        true_labels = np_states[i, max_window:len_path - max_window]
        pred_labels = jm.labels_.astype(int)
        score = balanced_accuracy_score(true_labels, pred_labels)
        balanced_accuracies.append(score)

    # Compute average balanced accuracy over all paths
    overall_accuracy = np.mean(balanced_accuracies)
    overall_std = np.std(balanced_accuracies)
    print(f"Average Balanced Accuracy over {nb_paths} paths: {overall_accuracy:.3f}")
    print(f"Overall Std over {nb_paths} paths: {overall_std:.3f}")
    # print(mean_per_cluster)


# TODO:
#  1) Aggiungi un'altra window e metti tutto in una funzione
#  1.5) Metti un boolean che tolga le sequenze con un solo state
#  1.7) C'è ancora il problema che se i labels riconosciuti sono solo di un tipo, allora ho dei NaN in mean e std
#  2) Comincia un uno script che prenda i dati reali di stocks
#  2.1) Implementa trans_cost_class





#     # # Set SOM Jump parameters
    # import torch
    # from sklearn.metrics import balanced_accuracy_score
    # from sklearn.preprocessing import StandardScaler
    # from SIGSOM.sigsom.utils.helpers import logger, SOMParams
    # from SIGSOM.sigsom.som.som_jump import SOM


    # # Set SOM Jump parameters
    # params = SOMParams(
    #     map_size=(2, 1),
    #     Tmax=3,
    #     Tmin=0.1,
    #     batch_size=512,
    #     iterations=5000,
    #     input_dims=all_features.shape[2],
    #     lr=0.01,
    #     lambda_jump=0.1,#.2,.3
    #     jump_penalty_epochs=10,
    #     decay="exponential",
    #     dump_path="./artifacts/simulation_experiments/"
    # )

    # balanced_accuracies = []

    # for i in range(nb_paths):
    #     # Scale each path (optional but often helpful)
    #     X_i = StandardScaler().fit_transform(all_features[i])
    #     X_i_torch = torch.tensor(X_i, dtype=torch.float32)

    #     som_model = SOM(logger=logger, params=params)
    #     som_model.initialize(X_i_torch)
    #     som_model.fit(X_train=X_i_torch)

    #     pred_labels = som_model.predict(X_i_torch).numpy()
    #     true_labels = np_states[i, max_window:len_path - max_window]

    #     # score = balanced_accuracy_score(true_labels, pred_labels)
    #     # balanced_accuracies.append(score)
    #     score = balanced_accuracy_score(true_labels, pred_labels)

    #     # If score is below 0.5, try flipping the predicted labels
    #     if score < 0.5:
    #         flipped_score = balanced_accuracy_score(true_labels, 1 - pred_labels)
    #         if flipped_score > score:
    #             pred_labels = 1 - pred_labels
    #             score = flipped_score

    #     balanced_accuracies.append(score)

    #     print(f"Path {i}: Balanced Accuracy = {score:.3f}")
    #     print(pred_labels)
    #     print(true_labels)

    # # Summary
    # overall_accuracy = np.mean(balanced_accuracies)
    # overall_std = np.std(balanced_accuracies)
    # print(f"\n🔎 Average Balanced Accuracy over {nb_paths} paths: {overall_accuracy:.3f}")
    # print(f"📉 Overall Std over {nb_paths} paths: {overall_std:.3f}")


# For length 250 , jp = 1, Average Balanced Accuracy over 1000 paths: 0.770, Overall Std over 1000 paths: 0.117
# For length 500 , jp = 1, Average Balanced Accuracy over 1000 paths: 0.789, Overall Std over 1000 paths: 0.072
# For length 1000 , jp = 1, Average Balanced Accuracy over 1000 paths: 0.800, Overall Std over 1000 paths: 0.040 
# For length 2000 , jp = 1, Average Balanced Accuracy over 1000 paths: 0.803, Overall Std over 1000 paths: 0.026

# Average Balanced Accuracy over 1000 paths: 0.805
# 📉 Overall Std over 1000 paths: 0.025