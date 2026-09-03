import torch
import torch.nn as nn
import torch.optim as optim

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
        
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.input_dims = input_dims
        if prototypes is not None:
            assert prototypes.shape == (self.n_prototypes, prototypes.shape[1]), "Prototypes shape mismatch."
            self.prototypes = nn.Parameter(torch.tensor(prototypes, dtype=torch.float32))
            print("im NOT initializing")
        else:
            self.prototypes = nn.Parameter(torch.Tensor(self.n_prototypes, input_dims))
            # self.sobol_initialization(data)
            kmeans = KMeans(n_clusters=self.n_prototypes, init='k-means++').fit(data)
            prototypes = kmeans.cluster_centers_
            self.prototypes.data.copy_(torch.tensor(prototypes, dtype=torch.float32))
            # print("im initializing")
            # nn.init.xavier_uniform_(self.prototypes.data)
        # Calculate and store the inverse covariance matrix
        # cov_matrix = np.cov(data.T)
        # self.cov_matrix_inv = torch.tensor(np.linalg.inv(cov_matrix), dtype=torch.float32)
        epsilon = 1e-6  # Regularization term
        cov_matrix = np.cov(data.T)
        cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
        self.cov_matrix_inv = torch.tensor(np.linalg.inv(cov_matrix), dtype=torch.float32)

    def forward(self, inputs):
        """
        Forward pass calculates the Mahalanobis distance between each input vector and each prototype.
        """
        # Shape: (batch_size, 1, input_dim) - Expand inputs for broadcasting
        input_expanded = inputs.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        
        # Shape: (1, n_prototypes, input_dim) - Expand prototypes for broadcasting
        prototypes_expanded = self.prototypes.unsqueeze(0)  # Shape: (1, n_prototypes, input_dim)
        
        # Shape: (batch_size, n_prototypes, input_dim) - Calculate difference
        diff = input_expanded - prototypes_expanded  # Shape: (batch_size, n_prototypes, input_dim)
        
        # Step 1: Multiply by the inverse covariance matrix
        # Shape: (batch_size, n_prototypes, input_dim)
        mahalanobis_term = torch.matmul(diff, self.cov_matrix_inv)
        
        # Step 2: Multiply by the transposed difference vector and sum over the last dimension
        # Shape: (batch_size, n_prototypes)
        distances = torch.sum(mahalanobis_term * diff, dim=2)
        # distances = torch.sqrt(distances)
        # Since distances are squared Mahalanobis distances, they should be non-negative.
        # If necessary, apply sqrt if you need the actual distance, but typically squared distances are fine.
        return distances

    def sobol_initialization(self, data):
        """
        Initializes prototypes using Sobol sequence.
        Arguments:
            data: Numpy array with shape (n_samples, latent_dim) representing the data to initialize the prototypes.
        """
        # Create SobolEngine for the given input dimensions
        sobol_engine = SobolEngine(dimension=self.input_dims, scramble=True)
        
        # Generate Sobol points and scale them to the range of the data
        sobol_points = sobol_engine.draw(self.n_prototypes)
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]
        sobol_points_scaled = data_min + sobol_points * (data_max - data_min)
        
        # Initialize prototypes with the scaled Sobol points
        self.prototypes.data.copy_(sobol_points_scaled)
        print("Sobol initialization complete")

    def kmeans_plus_plus_initialization(self, data):
        """
        Initializes prototypes using the k-means++ algorithm.
        Arguments:
            data: Numpy array with shape (n_samples, latent_dim) representing the data to initialize the prototypes.
        """
        kmeans = KMeans(n_clusters=self.n_prototypes, init='k-means++').fit(data)
        prototypes = kmeans.cluster_centers_
        self.prototypes.data.copy_(torch.tensor(prototypes, dtype=torch.float32))

# DESOM in pytorch
class SOMM(nn.Module):
    def __init__(self, logger, params):
        super(SOMM, self).__init__()
        self.logger = logger
        self.map_size = params.map_size
        self.input_dims = params.input_dims
        self.lr = params.lr
        self.n_prototypes = self.map_size[0] * self.map_size[1]
        self.model = None
        self.log_filename = 'som_log.csv'
        self.dump_path = params.dump_path
        self.dump_root_folder = os.path.join(self.dump_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.init_dump_folder()
        self.dict_log = dict()
        self.df_log = pd.DataFrame()
        # Early stopping
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

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
    
    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer.
        """
        self.eval()  
        with torch.no_grad():
            som_output = self.forward(x)
        index_of_bmu = som_output.min(dim=1)[1]  #Best Matching Unit
        return index_of_bmu

    @property
    def prototypes(self):
        return self.som_layer.prototypes

    def init_som_weights(self, X):
        indices = torch.randperm(X.size(0))[:self.som_layer.prototypes.size(0)]
        sample = X[indices]
        self.som_layer.prototypes.data.copy_(sample.data)

    def fit(self, X_train, 
            X_val=None,
            iterations=1000,  
            eval_interval=50,
            save_epochs=50, 
            batch_size=256, # 16 instead of 64
            Tmax=10, 
            Tmin=0.1, 
            decay='exponential'):
               
        self.logger.info(f"Save interval: {save_epochs}")

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val) if X_val is not None else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if X_val is not None else None

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        for epoch in range(iterations):

            T = Tmax * (Tmin / Tmax) ** (epoch / iterations) if decay == 'exponential' else Tmax - (Tmax - Tmin) * (epoch / iterations)

            self.train()
            for X_batch, in train_loader:
                optimizer.zero_grad()
                X_batch = X_batch
                som_output = self.forward(X_batch)
                bmu = som_output.min(dim=1)[1]
                dist = self.map_dist(bmu)  
                weights = self.neighborhood_function(dist, T)
                loss = self.compute_loss(weights, som_output)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                val_loss = self.validate_model(val_loader, T = T)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
                print(f'Epoch {epoch+1}, Training Loss: {loss}')
                self.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

                if epoch > 100:
                    self.early_stopping(val_loss, patience = 15,delta=0, verbose=False)

                if self.early_stop:
                    print(f"Early stopping at epoch{epoch} with validation error {val_loss}")
                    break

            if (epoch + 1) % eval_interval == 0:
                self.log_and_save_training_state(ite= epoch, loss =loss,weights=weights, d = som_output, y_pred = bmu, X_val=None, y_val_pred=None, T=None)
            
            if (epoch + 1) % save_epochs == 0:
                self.save_and_plot(ite = epoch, X_batch = X_batch, X_train = X_train, dump=True)

        self.save_and_plot(ite = epoch, X_batch = X_train, X_train = X_train, dump=True)
    
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
            return torch.exp(-(d ** 2) / (T ** 2))# change from 2 to 4
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