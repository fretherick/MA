# simulation_experiments

# Generate a sequence of observations from a hidden Markov model (HMM)
# Train a Jump Model on it

"""
Following the paper Nystrup, P., Lindström, E., & Madsen, H. (2020). Learning hidden Markov models with persistent
states by penalizing jumps, we simulate a Hidden Markov Model (HMM) and test a Jump Model over it.
"""

# -----Imports-----
import numpy as np
import os
import pandas as pd
import torch
import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
from jumpmodels.jumpmodels.jump import JumpModel
from other_models.hmm_model import HiddenMarkovModel
from other_models.hmm_utils import extract_hmm_features_vectorized
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import sys
import yaml
from munch import munchify 
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import balanced_accuracy_score
from SIGSOM.sigsom.utils.helpers import logger, SOMParams
from SIGSOM.sigsom.som.som_jump import SOM
from SIGSOM.sigsom.som.som import SOM
# ----- Load Params -----
import signatory
from sklearn.preprocessing import StandardScaler
from other_models.hmm_utils import extract_signature_features_centered




stage_name = "simulation_experiments"
stage_parameters = munchify(yaml.safe_load(open("params.yaml"))[stage_name])

dump_path = stage_parameters["output"]["dump_path"]
artifact_path = stage_parameters["output"]["artifact_path"]
metrics_dump_path = stage_parameters["output"]["metrics_dump_path"]

if not os.path.exists(dump_path):
    os.makedirs(dump_path)
if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
if not os.path.exists(metrics_dump_path):
        os.makedirs(metrics_dump_path)
# Load commons
common_params = munchify(yaml.safe_load(open("params.yaml"))["common"])
verbose = common_params["verbose"]
dump = common_params["dump"]
plot = common_params["plot"]
save_plots = common_params["save_plot"]


#---- Load HMM -----

# The paper HMM
def paper_hmm(hmm_params: dict) -> tuple[np.ndarray, np.ndarray]:
    hmm = HiddenMarkovModel(
        hmm_params["k"],
        np.array(hmm_params["initial_state_prob"]),
        np.array(hmm_params["transition_matrix"]),
        np.array(hmm_params["means"]),
        np.array(hmm_params["variances"]),
    )
    return hmm.simulate_paths(
        nb_paths=hmm_params["nb_paths"],
        T=hmm_params["len_path"],
        dt=hmm_params["dt"],
        more_states=hmm_params["more_states"]
    )

#---- Run the simulation -----

def main():
    hmm_params = stage_parameters["hmm"]["params"]
    sim_params = stage_parameters["simulation"]["params"]

    # … (everything up through generation of np_states, np_log_rets) …
    nb_paths = hmm_params["nb_paths"]
    len_path = hmm_params["len_path"]
    windows = sim_params["windows"]
    max_window = max(windows)
    nb_states = hmm_params["nb_states"]
    eff_timesteps = len_path - 2 * max_window

    # We no longer use OUT_FEAT = 9, since signature dimension depends on `depth`.
    depth = stage_parameters["signature"]["depth"]  # e.g. 2, 3, etc.
    # Compute D_sig for a 1‐dimensional time series at the chosen depth:
    D_sig = signatory.signature_channels(channels=1, depth=depth)

    # We'll build `all_features` of shape (nb_paths, eff_timesteps, len(windows) * D_sig)
    all_features = np.empty((nb_paths, eff_timesteps, len(windows) * D_sig), dtype=np.float32)

    # Generate HMM sample paths
    np_states, np_log_rets = paper_hmm(hmm_params=hmm_params)
    # np_log_rets has shape (nb_paths, len_path)

    # ——— Extract centered signature features for each window size ———
    # Let extract_signature_features_centered return an array of shape (nb_paths, T, D_sig),
    # where T == len_path. We then slice off the first/last max_window time steps so that
    # everything lines up (we only keep timesteps [max_window, …, len_path - max_window - 1]).
    for w_idx, window in enumerate(windows):
        # signature features over a sliding window of length 2*window + 1, centered at t
        sig_feats_full = extract_signature_features_centered(X=np_log_rets, l=window, depth=depth)
        # sig_feats_full.shape == (nb_paths, len_path, D_sig)

        # now remove the edges that were invalid (nan) because we only want [max_window : len_path - max_window]
        sig_feats_slice = sig_feats_full[:, max_window : (len_path - max_window), :]
        # sig_feats_slice.shape == (nb_paths, eff_timesteps, D_sig)

        # put into the appropriate slot in all_features
        start_col = w_idx * D_sig
        end_col = (w_idx + 1) * D_sig
        all_features[:, :, start_col:end_col] = sig_feats_slice
        print(all_features.shape)
    # Now `all_features` is (nb_paths, eff_timesteps, len(windows) * D_sig).
    # Standardize each path independently:
    for i in range(nb_paths):
        scaler = StandardScaler()
        all_features[i] = scaler.fit_transform(all_features[i])

    #---- Define the Clustering Model
    balanced_accuracies = []

    if stage_parameters["clustering"]["model"] == "som":
        # Define SOM parameters
        som_params = SOMParams(
            map_size = stage_parameters["clustering"]["som_jump"]["params"]["map_size"],
            Tmax= stage_parameters["clustering"]["som_jump"]["params"]["Tmax"],
            Tmin= stage_parameters["clustering"]["som_jump"]["params"]["Tmin"],
            batch_size= stage_parameters["clustering"]["som_jump"]["params"]["batch_size"],
            iterations= stage_parameters["clustering"]["som_jump"]["params"]["iterations"],
            input_dims= all_features.shape[2], #stage_parameters["clustering"]["som"]["params"]["input_dims"],
            lr= stage_parameters["clustering"]["som_jump"]["params"]["lr"],
            decay= stage_parameters["clustering"]["som_jump"]["params"]["decay"],
            dump_path= stage_parameters["clustering"]["som_jump"]["params"]["dump_path"],
        )
        # Run the som model
        for i in range(nb_paths):
            # Normalize features
            X_i = StandardScaler().fit_transform(all_features[i])
            # Turn to tensor (needed for the model)
            X_i_torch = torch.tensor(X_i, dtype=torch.float32)
            # Initlaize and fit the model
            som_model = SOM(logger=logger, params=som_params)
            som_model.initialize(X_i_torch)
            som_model.fit(X_train=X_i_torch)
            # Predict the labels
            pred_labels = som_model.predict(X_i_torch).numpy()
            true_labels = np_states[i, max_window:len_path - max_window]
            # Compute the balanced accuracy
            score = balanced_accuracy_score(true_labels, pred_labels)
            if score < 0.5:
                flipped_score = balanced_accuracy_score(true_labels, 1 - pred_labels)
                if flipped_score > score:
                    pred_labels = 1 - pred_labels
                    score = flipped_score

            balanced_accuracies.append(score)

    # # Jump Model
    if stage_parameters["clustering"]["model"] == "jump":
        jump_penalty = stage_parameters["clustering"]["jump_model"]["params"]["jump_penalty"]
        # LOOP over the rows of np_states and np_log_rets
        for i in range(nb_paths):
            print(all_features[i].shape)
            jm = JumpModel(n_components=nb_states, jump_penalty=jump_penalty, cont=False)
            jm.fit(all_features[i], np_log_rets[i], sort_by="cumret")
            # feat_centers[i] = jm.centers_
            # labels[i] = jm.labels_
            # mean_per_cluster[i] = jm.ret_
            # std_per_cluster[i] = jm.vol_
            # there is also jm.val_ which includes the best value for the objective function
            print(jm.labels_)
            print(np_states[i])
        # Compute balanced accuracy for each path
            true_labels = np_states[i, max_window:len_path - max_window]
            pred_labels = jm.labels_.astype(int)
            score = balanced_accuracy_score(true_labels, pred_labels)
            balanced_accuracies.append(score)

    # HMM
    if stage_parameters["clustering"]["model"] == "hmm":
        for i in range(nb_paths):
            log_ret_seq = np_log_rets[i].reshape(-1, 1)  # hmmlearn expects 2D array

            # Initialize HMM
            covariance_type = stage_parameters["clustering"]["hmm"]["params"]["covariance_type"]
            n_iter = stage_parameters["clustering"]["hmm"]["params"]["n_iter"]
            
            model = GaussianHMM(n_components=nb_states, covariance_type=covariance_type,init_params="", params="stmc",   n_iter=n_iter, random_state=42)
            # Initial state: always start in state 0
            model.startprob_ = np.array([1.0, 0.0])

            # Transition matrix with 0.9 self-transition
            model.transmat_ = np.array([
                [0.9, 0.1],
                [0.1, 0.9]
            ])

            # Means (μ₁ = μ₂ = 0)
            # D = all_features[i].shape[1]  # feature dimension
            # model.means_ = np.zeros((2, D))

            # # Variances (σ₁ = σ₂ = 0.01) – diagonal variances
            # model.covars_ = np.full((2, D), 0.01)

            model.means_ = np.array([[0.0], [0.0]])
            model.covars_ = np.array([[0.0001], [0.0001]])
            # Fit HMM to the log return sequence
            # model.fit(all_features[i])  # shape (224, D)
            # pred_labels = model.predict(all_features[i])

            model.fit(log_ret_seq)
            pred_labels = model.predict(log_ret_seq)

            # Evaluate accuracy
            true_labels = np_states[i, max_window:len_path - max_window]
            pred_labels_eval = pred_labels
            print(pred_labels_eval.shape)
            # pred_labels_eval = pred_labels[max_window:len_path - max_window]
            score = balanced_accuracy_score(true_labels, pred_labels_eval)
            if score < 0.5:
                flipped_score = balanced_accuracy_score(true_labels, 1 - pred_labels)
                if flipped_score > score:
                    pred_labels = 1 - pred_labels
                    score = flipped_score
            balanced_accuracies.append(score)

            print("model.means_.shape:", model.means_.shape)
            print("model.covars_.shape:", model.covars_.shape)
            # Save results
            labels[i] = pred_labels_eval
            # mean_per_cluster[i] = model.means_.flatten()  # shape: (n_states,)
            # std_per_cluster[i] = np.sqrt(np.array([np.diag(cov) for cov in model.covars_]))[:, 0]  # shape: (n_states,)
            # feat_centers[i] = model.means_  # full mean vectors (n_states, n_features)

            print(f"Path {i}: Balanced Accuracy = {score:.3f}")
            # print(pred_labels_eval)
            # print(true_labels)







    # Append accuracy and std
    overall_accuracy = np.mean(balanced_accuracies)
    overall_std = np.std(balanced_accuracies)
    print(f"\n🔎 Average Balanced Accuracy over {nb_paths} paths: {overall_accuracy:.3f}")
    print(f"📉 Overall Std over {nb_paths} paths: {overall_std:.3f}")
    # Dump the results
    if dump:
        metrics = {
            "balanced_accuracies": balanced_accuracies,
            "overall_accuracy": overall_accuracy,
            "overall_std": overall_std,
        }
        json_path = os.path.join(metrics_dump_path, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()



