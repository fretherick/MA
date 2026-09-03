import torch
import torch.nn as nn
import torch.optim as optim

# from metrics_pytorch import som_loss, kmeans_loss, quantization_error, topographic_error 
from SIGSOM.sigsom.utils.metrics_pytorch import som_loss, kmeans_loss, quantization_error, topographic_error 
from plotly.subplots import make_subplots
from csv import DictWriter
from ast import Not
import os
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import umap.umap_ as umap
import plotly.io as pio 
pio.templates.default = 'plotly_white'
# from torchviz import make_dot
from sklearn.cluster import KMeans
from torch.quasirandom import SobolEngine


class SOMLayer(nn.Module):
    """
    Self-Organizing Map layer class with rectangular topology

    # Example
    ```
        model.add(SOMLayer(map_size=(10,10)))
    ```
    # Arguments
        map_size: Tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1].
        prototypes: Numpy array with shape `(n_prototypes, latent_dim)` which represents the initial cluster centers
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`
    # Output shape
        2D tensor with shape: `(n_samples, n_prototypes)`
    """
    def __init__(self, map_size, input_dims, data, prototypes=None):
        super(SOMLayer, self).__init__()
        self.input_dims = input_dims
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.feature_weights = nn.Parameter(torch.ones(self.input_dims) / np.sqrt(self.input_dims))
        if prototypes is not None:
            assert prototypes.shape == (self.n_prototypes, prototypes.shape[1]), "Prototypes shape mismatch."
            self.prototypes = nn.Parameter(torch.tensor(prototypes, dtype=torch.float32))
            print("im NOT initializing")
        else:
            self.prototypes = nn.Parameter(torch.Tensor(self.n_prototypes, input_dims))
            nn.init.xavier_uniform_(self.prototypes.data)
    
    def forward(self, X):
        """
        Computes weighted features and distances to SOM prototypes.
        """
        weighted_X = X * torch.sqrt(torch.abs(self.feature_weights))  # Step 2(a) in Algorithm 1
        distances = torch.norm(weighted_X.unsqueeze(1) - self.prototypes.unsqueeze(0), dim=2)
        return distances

class SOMWithFeatureSelection(nn.Module):
    def __init__(self, logger, params):
        super(SOMWithFeatureSelection, self).__init__()
        self.logger = logger
        self.map_size = params.map_size
        self.input_dims = params.input_dims
        self.lr = params.lr
        self.n_prototypes = self.map_size[0] * self.map_size[1]
        self.l1_lambda = params.l1_lambda  # Regularization for sparsity
        self.iterations = params.iterations

        # Feature Weights Initialization
        # self.feature_weights = nn.Parameter(torch.ones(self.input_dims) / np.sqrt(self.input_dims))  # Step 1 in algorithm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def initialize(self,data):
        # Initialize SOM Layer
        self.som_layer = SOMLayer(self.map_size, self.input_dims, data)
        self.model = nn.ModuleDict({
            'som_layer': self.som_layer
        })
    
    def forward(self, x):
        som_output = self.som_layer(x)
        return som_output
    
    @property
    def prototypes(self):
        return self.som_layer.prototypes

    def init_som_weights(self, X):
        indices = torch.randperm(X.size(0))[:self.som_layer.prototypes.size(0)]
        sample = X[indices]
        self.som_layer.prototypes.data.copy_(sample.data)

    
    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes
        (rectangular grid topology) using NumPy.

        Arguments:
            y_pred: Tensor with shape (n_samples,) containing cluster assignments as indices.

        Returns:
            A NumPy array containing the pairwise Manhattan distances between assigned cell of 
            data point i and cell k on the map.

        keras code:
            labels = np.arange(self.n_prototypes)
            tmp = np.expand_dims(y_pred, axis=1)
            d_row = np.abs(tmp - labels) // self.map_size[1]
            d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
            return d_row + d_col
        """
        y_pred = y_pred.to(self.device) if not y_pred.is_cuda else y_pred
        labels = torch.arange(self.n_prototypes, device=self.device)
        labels = labels.view(1, -1)
        y_pred = y_pred.view(-1, 1)

        d_row = torch.abs(y_pred // self.map_size[1] - labels // self.map_size[1])
        d_col = torch.abs(y_pred % self.map_size[1] - labels % self.map_size[1])

        return (d_row + d_col).float().to(self.device)


    @staticmethod
    def neighborhood_function(d, T, neighborhood='gaussian'):
        """
        SOM neighborhood function using PyTorch for potential GPU acceleration.

        Arguments:
            d (torch.Tensor): Distance on the map.
            T (float or torch.Tensor): Temperature parameter controlling the spread.
            neighborhood (str): Type of neighborhood function, 'gaussian' or 'window'.

        Returns:
            torch.Tensor: Neighborhood weight.
        """
        if neighborhood == 'gaussian':
            return torch.exp(-(d ** 2) / (T ** 2))
        elif neighborhood == 'window':
            return (d <= T).float()

    def validate_model(self, val_loader, T):
        self.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, in val_loader:
                X_val_batch = X_val_batch.to(self.device)
                som_output_val = self.forward(X_val_batch)
                y_val_pred = som_output_val.min(dim=1)[1]
                dist_val = self.map_dist(y_val_pred)
                weights_val = self.neighborhood_function(dist_val, T)
                val_loss = self.compute_loss(weights_val, som_output_val)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        return average_val_loss

    def compute_loss(self, weights, distances):
        som_loss_value = som_loss(weights, distances)
        total_loss = som_loss_value
        return total_loss

    def early_stopping(self,val_loss, patience, delta=0, verbose=False):
        if self.best_loss is None or val_loss < self.best_loss - delta:
            self.best_loss = val_loss
            self.save_best_model_weights()
            self.counter = 0
        else:
            self.counter += 1
            if verbose:
                print(f'EarlyStopping counter: {self.counter} out of {patience}')
            if self.counter >= patience:
                self.early_stop = True

    def save_and_plot(self, ite, X_batch, X_train, dump=True):
        self.save_model_weights(iteration=ite)
        self.plot_loss_functions(dump=dump, iteration=ite)
        self.plot_errors(dump=dump, iteration=ite)
    def log_and_save_training_state(self, ite, loss,weights, d, y_pred, X_val=None, y_val_pred=None, T=None):
        """
        Handles logging metrics and saving state during training.

        Args:
        ite (int): Current iteration or epoch number.
        loss (float): Current loss value.
        d (array): Current distance metrics from SOM or similar.
        y_pred (array): Predicted labels or similar metrics.
        X_val (array, optional): Validation dataset. Used for validation metrics if provided.
        y_val_pred (array, optional): Predicted labels for the validation dataset.
        T (float, optional): Current temperature in training (for algorithms that use annealing or similar processes).
        """
        # Initialize or update log dictionary
        self.dict_log = {'iter': ite, 'T': T if T is not None else ''}
        self.dict_log['L'] = float(loss)
        # These 3 below all show 0
        self.dict_log['Lkm'] = kmeans_loss(y_pred, d)
        print(self.dict_log['Lkm'])
        self.dict_log['latent_quantization_err'] = quantization_error(d)
        self.dict_log['latent_topographic_err'] = topographic_error(d, self.map_size)
        self.dict_log["som_loss"] = som_loss(weights,d)
        decimal_places = 3
        factor = 10 ** decimal_places
        # Log the information
        self.logger.info(f"Iteration {ite} - T={torch.round(T, 3) if T else 'N/A'}")
        self.logger.info(f"  [Train] - L = {round(self.dict_log['L'],3)}, L_km = {torch.round(self.dict_log['Lkm']*factor)/factor}")
        self.logger.info(f"  [Train] - L = {round(self.dict_log['L'],3)}, som_loss = {torch.round(self.dict_log['som_loss']*factor)/factor}")
        # Handle logging to file
        if ite == 0:
            self.init_log_writer(fieldnames=list(self.dict_log.keys()))
        else:
            self.write_log_record(fieldnames=list(self.dict_log.keys()))

        # Append the current log to the DataFrame, creating it if it doesn't exist
        new_df_log = pd.DataFrame([self.dict_log]) # change from the original pd.attach doesnt exist so we need concat therefore creat a new df first and concat witht he old one 
        if hasattr(self, 'df_log'):
            self.df_log = pd.concat([self.df_log, new_df_log], ignore_index=True)  
        else:
            self.df_log = pd.DataFrame([self.dict_log])

    def cluster_in_latent_space(self, x):
        self.eval()  
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        with torch.no_grad():
            som_output = self(x)  
            _, indices = torch.min(som_output, dim=1) 

        return indices

    def init_dump_folder(self):

        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

        if not os.path.exists(self.dump_root_folder):
            os.makedirs(self.dump_root_folder)

    def check_folder_exists(self, path, create=False):

        if not os.path.exists(path):
            if create:
                os.makedirs(path)
                b_exists = True
            else:
                b_exists = False

        else:
            b_exists = True

        return b_exists

    def get_dump_path(self, artifact=None):

        dump_path = self.dump_root_folder

        if artifact is None:
            pass

        elif artifact == 'model':
            dump_path = os.path.join(dump_path, 'model')

        elif artifact == 'log':
            dump_path = os.path.join(dump_path, 'log')

        elif artifact == 'loss':
            dump_path = os.path.join(dump_path, 'loss')

        elif artifact == 'error':
            dump_path = os.path.join(dump_path, 'error')

        elif artifact == 'som':
            dump_path = os.path.join(dump_path, 'som')

        else:
            self.logger.warning(f"Artifact type {artifact} not recognized. Returning root folder.")

        _ = self.check_folder_exists(dump_path, create=True)

        return dump_path

    def init_log_writer(self, fieldnames: list) -> None:
        save_dir = self.get_dump_path(artifact='log')
        with open(os.path.join(save_dir, self.log_filename), 'w') as csvfile:
            self.log_file = DictWriter(csvfile, fieldnames=fieldnames)
            self.log_file.writeheader()
    def write_log_record(self, fieldnames: list):
        save_dir = self.get_dump_path(artifact='log')
        with open(os.path.join(save_dir, self.log_filename), 'a') as csvfile:
            self.log_file = DictWriter(csvfile, fieldnames=fieldnames)
            self.log_file.writerow(self.dict_log)

    def save_model_weights(self, iteration: int):
        filename = f"iter_{iteration}_model_weights.pth" if iteration else "model_weights.pth"
        save_dir = self.get_dump_path(artifact='model')
        torch.save(self.state_dict(), os.path.join(save_dir, filename))

    def save_best_model_weights(self):
        filename = "best_model_weights.pth"
        save_dir = self.get_dump_path(artifact='model')
        torch.save(self.state_dict(), os.path.join(save_dir, filename))

    def plot_loss_functions(self, dump=False, iteration=None):
        _df_log = self.df_log

        fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {}]], shared_xaxes=True, horizontal_spacing=0.02,
                            vertical_spacing=0.02)

        fig.add_trace(
            go.Scatter(x=_df_log['iter'], y=_df_log['L'], line=dict(color='black'), line_shape='linear', name='total'),
            row=1, col=1)
        fig.update_xaxes(title='iteration', row=1, col=1)
        fig.update_yaxes(title='loss | training set', range=[0.0, 1.0], row=1, col=1)

        fig.add_trace(go.Scatter(x=_df_log['iter'], y=_df_log['Lkm'], line=dict(color='slategrey'), line_shape='linear',
                                 name='kmean'), row=2, col=1)
        fig.update_xaxes(title='iteration', row=2, col=1)
        fig.update_yaxes(title='loss som | training set', range=[0.0, 1.0], row=2, col=1)

        if 'L_val' in _df_log.columns:
            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=_df_log['Lr_val'], line=dict(color='slategrey'), line_shape='linear',
                           name='reconstruction'), row=1, col=2)
            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=_df_log['Lsom_val'], line=dict(color='lightgrey'), line_shape='linear',
                           name='som'), row=1, col=2)
            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=(_df_log['L_val']), line=dict(color='black'), line_shape='linear',
                           name='total'), row=1, col=2)
            fig.update_xaxes(title='iteration', row=1, col=2)
            fig.update_yaxes(title='loss | validation set', range=[0.0, 1.0], row=1, col=2)

            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=_df_log['Lkm_val'], line=dict(color='slategrey'), line_shape='linear',
                           name='kmean'), row=2, col=2)
            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=_df_log['Ltop_val'], line=dict(color='lightgrey'), line_shape='linear',
                           name='topographic'), row=2, col=2)
            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=_df_log['Lsom_val'], line=dict(color='black'), line_shape='linear',
                           name='som'), row=2, col=2)
            fig.update_xaxes(title='iteration', row=2, col=2)
            fig.update_yaxes(title='loss som | validation set', range=[0.0, 1.0], row=2, col=2)

        if dump:
            if iteration:
                filename = f"iter_{iteration}_loss_functions.png"
            else:
                filename = f"loss_functions.png"

            save_dir = self.get_dump_path(artifact='loss')
            fig.write_image(os.path.join(save_dir, filename), width=1920, height=1080)
        else:
            fig.show()

    def plot_errors(self, dump=False, iteration=None):
        _df_log = self.df_log
        
        fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {}]], shared_xaxes=True, horizontal_spacing=0.02,
                            vertical_spacing=0.02)
        
        fig.add_trace(go.Scatter(x=_df_log['iter'], y=_df_log['latent_quantization_err'].apply(lambda x: x.detach().numpy()), line=dict(color='black'),
                                 line_shape='linear', name='reconstruction'), row=1, col=1)
        fig.update_xaxes(title='iteration', row=1, col=1)
        fig.update_yaxes(title='latent quantization error | training set', row=1, col=1)
        
        fig.add_trace(go.Scatter(x=_df_log['iter'], y=_df_log['latent_topographic_err'], line=dict(color='black'),
                                 line_shape='linear', name='som'), row=2, col=1)
        fig.update_xaxes(title='iteration', row=2, col=1)
        fig.update_yaxes(title='latent topographic error | training set', row=2, col=1)

        if 'latent_quantization_err_val' in _df_log.columns:
            fig.add_trace(
                go.Scatter(x=_df_log['iter'], y=_df_log['latent_quantization_err_val'], line=dict(color='black'),
                           line_shape='linear', name='reconstruction'), row=1, col=2)
            fig.update_xaxes(title='iteration', row=1, col=2)
            fig.update_yaxes(title='latent quantization error | validation set', row=1, col=2)

            fig.add_trace(go.Scatter(x=_df_log['iter'], y=_df_log['latent_topographic_err'], line=dict(color='black'),
                                     line_shape='linear', name='som'), row=2, col=2)
            fig.update_xaxes(title='iteration', row=2, col=2)
            fig.update_yaxes(title='latent topographic error | validation set', row=2, col=2)

        fig.update(layout_showlegend=False)

        if dump:
            if iteration:
                filename = f"iter_{iteration}_errors.png"
            else:
                filename = f"errors.png"

            save_dir = self.get_dump_path(artifact='error')
            fig.write_image(os.path.join(save_dir, filename), width=1920, height=1080)
        else:
            fig.show()

    ########### JUMP Part ############

    def refine_bmu_sequence(self, bmu_initial, X_train, lambda_jump=0.1):
        """
        Step 2: Adjust BMU sequence using Dynamic Programming (Bellman equation).
        
        Args:
            bmu_initial (torch.Tensor): Initial BMU assignments from training.
            X_train (torch.Tensor): Training data.
            lambda_jump (float): Strength of the jump penalty.
        
        Returns:
            smoothed_bmu (torch.Tensor): Optimized BMU sequence.
        """

        T = len(bmu_initial)  # Number of time steps
        num_units = self.n_prototypes  # SOM lattice size

        # Step 1: Compute Distance Cost Matrix
        distance_cost = torch.zeros((T, num_units), device=self.device)

        for t in range(T):
            for i in range(num_units):
                distance_cost[t, i] = torch.norm(X_train[t] - self.prototypes[i])  # Use SOM prototypes

        # Step 2: Initialize DP Table for Minimum Cost
        dp = torch.full((T, num_units), float('inf'), device=self.device)
        dp[0] = distance_cost[0]  # First time step uses only distance cost

        # Step 3: Track Best Previous BMU for Backtracking
        prev_bmu = torch.zeros((T, num_units), dtype=torch.long, device=self.device)

        # Step 4: Fill DP Table Using Bellman Recursion
        for t in range(1, T):
            for i in range(num_units):
                min_cost = float('inf')
                best_prev_bmu = 0

                for j in range(num_units):  # Iterate over all previous BMUs
                    transition_cost = lambda_jump * (i - j) ** 2  # L2 jump penalty
                    total_cost = dp[t-1, j] + distance_cost[t, i] + transition_cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev_bmu = j

                dp[t, i] = min_cost
                prev_bmu[t, i] = best_prev_bmu  # Store best previous BMU for backtracking

        # Step 5: Backtrack to Retrieve the Best BMU Sequence
        smoothed_bmu = torch.zeros(T, dtype=torch.long, device=self.device)
        smoothed_bmu[-1] = torch.argmin(dp[-1])  # Start from last time step

        for t in range(T - 2, -1, -1):  # Backtrack through DP table
            smoothed_bmu[t] = prev_bmu[t + 1, smoothed_bmu[t + 1]]

        return smoothed_bmu


    ############# Updated Fit funciton ############

    def fit(self, X_train, 
            iterations=1000, 
            jump_penalty_epochs=10,  
            lambda_jump=0.1, 
            batch_size=64, 
            Tmax=10, 
            Tmin=0.1, 
            decay='exponential'):
        
        self.logger.info(f"Using jump penalty with {jump_penalty_epochs} iterations")

        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(iterations):
            
            # Step 1: Train SOM normally
            T = Tmax * (Tmin / Tmax) ** (epoch / iterations) if decay == 'exponential' else Tmax - (Tmax - Tmin) * (epoch / iterations)
            self.train()

            for X_batch, in train_loader:
                optimizer.zero_grad()
                som_output = self.forward(X_batch)
                bmu = som_output.min(dim=1)[1]  # Get BMU indices
                dist = self.map_dist(bmu)  
                weights = self.neighborhood_function(dist, T)
                loss = self.compute_loss(weights, som_output)
                loss.backward()
                optimizer.step()

            # Step 2: Apply BMU Smoothing Every 'jump_penalty_epochs' Iterations
            if (epoch + 1) % jump_penalty_epochs == 0:
                with torch.no_grad():
                    full_som_output = self.forward(X_train)  # Compute full dataset's SOM output
                    full_bmu = full_som_output.min(dim=1)[1]  # Compute BMUs for full dataset

                smoothed_bmu = self.refine_bmu_sequence(full_bmu, X_train, lambda_jump=lambda_jump)
            # Update Feature Weights Every Few Iterations
            # if (epoch + 1) % jump_penalty_epochs == 0:
            #     # smoothed_bmu = self.refine_bmu_sequence(bmu, X_train, lambda_jump=lambda_jump)
            #     self.update_feature_weights(X_batch, bmu)
            if (epoch + 1) % jump_penalty_epochs == 0:
                with torch.no_grad():
                    full_som_output = self.forward(X_train)  # Compute SOM output for full dataset
                    full_bmu = full_som_output.min(dim=1)[1]  # Get BMUs for entire dataset

                smoothed_bmu = self.refine_bmu_sequence(full_bmu, X_train, lambda_jump=lambda_jump)

                # Now, update feature weights using the full BMU sequence
                self.update_feature_weights(X_train, smoothed_bmu)


        print("Training with jump penalty completed.")

    def predict(self, X):
        """
        Predicts the Best Matching Unit (BMU) for input data using the trained SOM,
        incorporating feature weighting and the jump penalty via dynamic programming.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            smoothed_bmu (torch.Tensor): The optimal BMU sequence with jump penalty applied.
        """
        self.eval()
        X = X.to(self.device)

        with torch.no_grad():
            # Apply feature weights before computing BMU
            weighted_X = X * torch.sqrt(torch.abs(self.som_layer.feature_weights))
            weighted_prototypes = self.prototypes * torch.sqrt(torch.abs(self.som_layer.feature_weights))
            
            # Compute weighted distances
            distances = torch.norm(weighted_X.unsqueeze(1) - weighted_prototypes.unsqueeze(0), dim=2)
            
            # Get BMU indices
            bmu_initial = distances.min(dim=1)[1]  

        # Apply dynamic programming to refine the BMU sequence
        smoothed_bmu = self.refine_bmu_sequence(bmu_initial, weighted_X, lambda_jump=0.1)  # Adjust lambda_jump as needed

        return smoothed_bmu


    def greedy_online_prediction(self, X, lambda_jump=0.1):
        """
        Implements greedy online state classification for sequential SOM prediction.

        Args:
            X (torch.Tensor): Input time series data (shape: [T, input_dim]).
            lambda_jump (float): Jump penalty factor.

        Returns:
            predicted_bmus (torch.Tensor): Greedy estimated BMU sequence.
            arrival_costs (torch.Tensor): Updated arrival costs for each time step.
        """
        T = X.shape[0]  # Number of time steps
        num_units = self.n_prototypes  # Total number of BMUs

        # Initialize arrival cost A_{t-1}
        arrival_cost = torch.zeros(num_units, device=self.device)

        # Store predicted BMUs
        predicted_bmus = torch.zeros(T, dtype=torch.long, device=self.device)
        arrival_costs = torch.zeros(T, num_units, device=self.device)

        for t in range(T):
            # Compute quantization loss for all BMUs
            # quantization_loss = torch.norm(X[t] - self.prototypes, dim=1)  # Shape: (num_units,)
            weighted_X = X[t] * torch.sqrt(torch.abs(self.som_layer.feature_weights))
            weighted_prototypes = self.prototypes * torch.sqrt(torch.abs(self.som_layer.feature_weights))
            quantization_loss = torch.norm(weighted_X - weighted_prototypes, dim=1)  # Shape: (num_units,)
            if t == 0:
                # First step: no jump penalty
                arrival_cost = quantization_loss
            else:
                prev_bmu = predicted_bmus[t - 1]

                # Compute transition cost for all BMUs (shape: [num_units])
                transition_cost = lambda_jump * (torch.arange(num_units, device=self.device) != prev_bmu).float()

                # Update arrival cost using the previous step's arrival cost
                # arrival_cost = torch.min(
                #     arrival_cost.view(1, -1) + transition_cost.view(-1, 1), dim=1
                # )[0] + quantization_loss  # Min over previous BMUs
                arrival_cost = torch.min(arrival_cost.view(1, -1) + transition_cost.view(-1, 1) + quantization_loss.view(-1, 1), dim=1)[0]
            # Store arrival cost for visualization
            arrival_costs[t] = arrival_cost

            # Compute estimated state s_t (Eq. 2)
            predicted_bmus[t] = torch.argmin(quantization_loss + arrival_cost)

        return predicted_bmus, arrival_costs
    
    def update_feature_weights(self, X, smoothed_bmu):
        """
        Applies soft thresholding to update feature weights while handling numerical stability issues.
        """
        T = X.shape[0]

        print("Shape of X:", X.shape)
        print("Shape of smoothed_bmu:", smoothed_bmu.shape)
        print("Shape of prototypes:", self.prototypes.shape)

        # Check BMU index validity
        if smoothed_bmu.max() >= self.n_prototypes or smoothed_bmu.min() < 0:
            raise ValueError(f"Invalid BMU indices detected! Expected range: 0-{self.n_prototypes-1}, got {smoothed_bmu}")

        # Initialize feature gradients
        feature_grad = torch.zeros_like(self.som_layer.feature_weights, device=self.device)

        for t in range(T):
            assigned_prototype = self.prototypes[smoothed_bmu[t]]

            # Check for NaNs or Inf in X or assigned prototype
            if torch.isnan(X[t]).any() or torch.isnan(assigned_prototype).any():
                print(f"NaN detected at time step {t} in input X or assigned prototype, skipping update for this step.")
                continue  

            feature_grad += (X[t] - assigned_prototype) ** 2  

        # Prevent division by zero
        feature_grad /= max(T, 1e-6)  

        # Apply soft-thresholding (L1 sparsity)
        new_feature_weights = torch.sign(feature_grad) * torch.clamp(torch.abs(feature_grad) - self.l1_lambda, min=0)

        # Check for NaNs or Inf before updating
        if torch.isnan(new_feature_weights).any() or torch.isinf(new_feature_weights).any():
            print("NaN or Inf detected in new feature weights, skipping update.")
            return

        # Update feature weights safely
        clamped_weights = torch.clamp(self.som_layer.feature_weights.data, min=1e-6, max=1.0)

        # Explicitly set very small values to zero
        clamped_weights[clamped_weights == 1e-6] = 0.0

        self.som_layer.feature_weights.data.copy_(clamped_weights)
        # self.som_layer.feature_weights.data.copy_(new_feature_weights)

        # Fix: Explicitly assign clamped values instead of in-place operation
        self.som_layer.feature_weights.data.copy_(
            torch.clamp(self.som_layer.feature_weights.data, min=1e-6, max=1.0)
        )

        print("Feature weights updated successfully.")