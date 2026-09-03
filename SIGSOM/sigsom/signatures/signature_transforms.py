# Imports
import torch
from typing import List, Tuple
from dataclasses import dataclass
import signatory
import numpy as np
import torch
import pickle
import os
from pathlib import Path
import bz2
import pandas as pd
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math
from sympy import mobius, divisors

############################## Start of Data Loading ##############################
def get_first_file(data_path):
    for file_path in data_path.iterdir():
        if file_path.is_file():
            # Here you can process each dataset (file)
            # print(f"Processing {file_path.name}")
            # Example of loading data if it's a CSV
            # import pandas as pd

            first_file = next((file_path for file_path in data_path.iterdir() if file_path.is_file()), None)
    return first_file

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)   
      
def load_pbz2(path):
    with bz2.open(path, 'rb') as f:
        data = pickle.load(f)
    return data
############################## End of Data Loading ##############################

############################## Start of Augmentations ##############################
# hepler function to get time augmentaiton for lead lags
def get_time_vector(size: int, length: int) -> torch.Tensor:
    """
    Helper function for the lead lag transforms 
    # size number of intervals, how many timestamps between 0 and 1
    """
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)

# Time augmentation
def time_transform(x: torch.Tensor, steps= "scale") -> torch.Tensor:
    if steps== "scale":
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    if steps == "normal":
        t = torch.arange(0, x.shape[1]).reshape(1, -1, 1).repeat(x.shape[0], 1, 1).to(x.device)
    return torch.cat([t,x], dim = 2)

# Time embedding
def time_embedding(x: torch.Tensor, steps= "scale") -> torch.Tensor:
    if steps== "scale":
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    if steps == "normal":
        t = torch.arange(0, x.shape[1]).reshape(1, -1, 1).repeat(x.shape[0], 1, 1).to(x.device)

    t_rep = torch.repeat_interleave(t, repeats=2, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    # concat the data, it will (almost) tripe in size
    x_ll = torch.cat([
        t_rep[:, 1:],
        x_rep[:, 0:-1],
    ], dim=2)
    return x_ll

# Basepoint augmentation
# already part of the singature function , basepoint = True

# Lead Lag : input (windows, sequence length, feature dimension) output (windwos, lead_lag dim (= double feature dim -1), [lead, lag])
def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate path (two-dimensional tensor).
    """
    # Repeat each sample twice: dim(samplesize, fetures) -> dim(2*samplesize, features) 
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    
    # Interleave repeated values to produce lead-lag transformation
    lead = x_rep[:, 1:]
    lag = x_rep[:, :-1]
    x_ll = torch.cat((lead, lag), dim=2)
    
    return x_ll

# Lead lag with time: input (windows, sequence length, feature dimension) output (windwos, lead_lag dim (= double feature dim -1), [time(lead),lead, lag])
def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    # gives back time interval between 0 adn 1 with the size of (batch/window, sequence length, 1)
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    # we need 3 replicants of all the data for what we want to do
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=1)
    # concat the data, it will (almost) tripe in size
    x_ll = torch.cat([
        t_rep[:, 2:],
        x_rep[:, 1:-1],
        x_rep[:, :-2],
    ], dim=2)
    return x_ll
 
# Cat Lags
def cat_lags(x: torch.Tensor, m: int) -> torch.Tensor:
    q = x.shape[1]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(q-m+1):
        x_lifted.append(x[:, i:i + m])
        x_lifted[-1] = x_lifted[-1].view(x_lifted[-1].shape[0], 1, x_lifted[-1].shape[1]*x_lifted[-1].shape[2])
    return torch.cat(x_lifted, dim=1).to(x.device)  

# Cumulative sum
def cum_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim # what is the appropriate dim, depends if we use batch or not without its dim 0 with its dim 1
    return x.cumsum(dim=dim).to(x.device)

# stream preserving augmentation
# maybe implement as well

############################## End of Augmentations ##############################

############################## Start of Winodws ##############################
# works on a shape(x,y) tensor
def rolling_window(x: torch.Tensor, window: int) -> torch.Tensor:
    N = x.shape[0] // window
    result = torch.empty((N, window, x.shape[1]))

    for i in range(N):
        result[i, :, :] = x[i*window: window*(i+1),:]

    return result
# works for a tensor of shape (x,y,z)
def rolling_window_on_window(x: torch.Tensor, window: int) -> torch.Tensor:
    N = x.shape[1] // window
    result = torch.empty((x.shape[0],N, window, x.shape[2]))

    for i in range(N):
        result[:, i, :, :] = x[:,i*window: window*(i+1),:]

    return result
# for a s continuous widnow with no time gaps
def rolling_window1(x: torch.Tensor, window: int) -> torch.Tensor:
    N = (x.shape[0]-1) // (window-1)
    result = torch.empty((N, window, x.shape[1]))
    for i in range(N):
        if i == 0:
            result[i, :, :] = x[0:window, :]
        else:
            result[i, :, :] = x[(i*(window))-i: (window*(i+1))-i,:]

    return result

def expanding_window(tensor, x_lag):
    if (x_lag > tensor.shape[1]):
        raise ValueError('The lag is larger than the sequence %s.' % x_lag)
    
    sections, seq_length, feature_dim = tensor.shape
    expanded_windows = []

    for section in range(sections):
        section_windows = []
        for i in range(seq_length//x_lag + 1):
            window = tensor[section, :(1+i)*x_lag, :]
            section_windows.append(window)
        expanded_windows.append(section_windows)

    max_seq_length = seq_length
    padded_windows = []
    for section_windows in expanded_windows:
        padded_section = []
        for window in section_windows:
            padding = torch.zeros((max_seq_length - window.shape[0], feature_dim), dtype=tensor.dtype)
            padded_window = torch.cat((window, padding), dim=0)
            padded_section.append(padded_window.unsqueeze(0))
        padded_windows.append(torch.cat(padded_section))

    result = torch.stack(padded_windows)
    return result

# def dyadic_window(x, dyadic_depth):

#     windows = []
#     n = x.shape[1]
#     x = x.unsqueeze(1) 

#     max_depth = int(np.floor(np.log2(n/2))) # ensure each window has at least 2 elements
#     if dyadic_depth > max_depth:
#         raise ValueError('The depth chosen is too high. Maximum allowed depth is %s.' % max_depth)

#     for i in range(dyadic_depth-1,-1,-1):
#         denominator = 2 ** (dyadic_depth - i)
#         window_length = n // denominator
#         windows.append(torch.cat([x[:,:, t:t + window_length] for t in range(0, n-window_length+1, window_length)], dim=1))

#     return windows

def dyadic_window(x, dyadic_depth):
    windows = []
    n = x.shape[1]
    x = x.unsqueeze(1)  

    max_depth = int(np.floor(np.log2(n/2)))  
    if dyadic_depth > max_depth:
        raise ValueError('The depth chosen is too high. Maximum allowed depth is %s.' % max_depth)

    windows.append(x)

    for i in range(dyadic_depth-1,-1,-1):
        denominator = 2 ** (dyadic_depth - i)
        window_length = n // denominator
        windows.append(torch.cat([x[:,:, t:t + window_length] for t in range(0, n-window_length+1, window_length)], dim=1))

    return windows

############################## End of Windows ##############################

############################## Start of Scaling ##############################
# scaling this is a standard scaler
class StandardScalerTS:
    """ Standard scales a given (indexed) input vector along the specified axis. """
    def __init__(self, axis=(1,)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)

        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
    
        return x * self.std.to(x.device) + self.mean.to(x.device)
# pre and post scalers
###### For Multivariate TS
def num_terms_at_depth(log_or_normal,d, n):
    """
    Compute the number of terms in the log signature at depth n using the given formula.
    
    Args:
    - d: Dimension of the input path.
    - n: Depth level.

    Returns:
    - Number of log signature terms at depth n.
    """
    if log_or_normal == "log":
        result = 0
        for m in divisors(n):
            result += mobius(m) * d**(n // m)
        return result // n
    elif log_or_normal == "normal":
        return d ** n
    else:
        raise ValueError("Invalid input for log_or_normal. Must be 'log' or 'normal'.")

def rescale_signature_by_factorial(log_or_normal, signature, dimension, depth):
    """
    Multiplies the terms in the signature by the factorial of their respective depth level.
    Args:
    - signature: The signature tensor.
    - dimension: The number of input dimensions.
    - depth: The maximum depth of the signature.

    Returns:
    - Rescaled signature (factorial-weighted by depth level).
    """
    rescaled_signature = []
    current_index = 0

    for d in range(1, depth + 1):
        # Calculate the number of terms at this depth level
        n_terms_at_depth_d = num_terms_at_depth(log_or_normal, dimension, d)
        # print(n_terms_at_depth_d)
        # print(dimension,d)
        # Compute factorial of the current depth level
        depth_factorial = math.factorial(d)

        # Extract the terms for the current depth and multiply by the depth factorial
        terms_at_depth_d = signature[..., current_index:current_index + n_terms_at_depth_d] * depth_factorial

        # Append the rescaled terms to the rescaled_signature list
        rescaled_signature.append(terms_at_depth_d)

        # Update the index to point to the next depth section
        current_index += n_terms_at_depth_d

    # Concatenate rescaled terms along the feature dimension
    return torch.cat(rescaled_signature, dim=-1)

# copy form the other file






# maybe implement, not really shown to improve results see generalzied path signature paper

############################## End of Scaling ##############################

############################## Start of Signatures Transfrom ##############################
def signature(x: torch.Tensor, depth: int, basepoint = False):
    return signatory.signature(x, depth=depth, basepoint=basepoint)

def log_singature(x: torch.Tensor, depth: int, basepoint = False):
    return signatory.logsignature(x, depth=depth, basepoint=basepoint)
############################## Start of Signatures Transfrom ##############################

########################### Augmentation on 4 dimensional input (window, subwindow, seq length, features) ###########################

# hepler function to get time augmentaiton for lead lags
def get_time_vector4(split: int, size: int, length: int) -> torch.Tensor:
    """
    Helper function for the lead lag transforms 
    # size number of intervals, how many timestamps between 0 and 1
    """
    return torch.linspace(0, 1, length).reshape(1,1, -1, 1).repeat(1, size, 1, 1).repeat(split, 1,1,1)

# Time augmentation
def time_transform4(x: torch.Tensor, steps= "scale") -> torch.Tensor:
    if steps== "scale":
        t = get_time_vector4(x.shape[0], x.shape[1],x.shape[2]).to(x.device)
    if steps == "normal":
        t = torch.arange(0, x.shape[2]).reshape(1, 1, -1, 1).repeat(1,x.shape[1], 1, 1).repeat(x.shape[0],1, 1, 1).to(x.device)
    return torch.cat([t,x], dim = 3)

# Time embedding
def time_embedding4(x: torch.Tensor, steps= "scale") -> torch.Tensor:
    if steps== "scale":
        t = get_time_vector(x.shape[0], x.shape[1],x.shape[2]).to(x.device)
    if steps == "normal":
        t = torch.arange(0, x.shape[2]).reshape(1, 1, -1, 1).repeat(1,x.shape[1], 1, 1).repeat(x.shape[0],1, 1, 1).to(x.device)

    t_rep = torch.repeat_interleave(t, repeats=2, dim=2)
    x_rep = torch.repeat_interleave(x, repeats=2, dim=2)
    # concat the data, it will (almost) tripe in size
    x_ll = torch.cat([
        t_rep[:,:, 1:],
        x_rep[:,:, 0:-1],
    ], dim=3)
    return x_ll

# Basepoint augmentation
# already part of the singature function , basepoint = True

# Lead Lag : input (windows, sequence length, feature dimension) output (windwos, lead_lag dim (= double feature dim -1), [lead, lag])
def lead_lag_transform4(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate path (two-dimensional tensor).
    """
    # Repeat each sample twice: dim(samplesize, fetures) -> dim(2*samplesize, features) 
    x_rep = torch.repeat_interleave(x, repeats=2, dim=2)
    
    # Interleave repeated values to produce lead-lag transformation
    # lead = x_rep[:, 1:]
    # lag = x_rep[:, :-1]
    # x_ll = torch.cat((lead, lag), dim=3)
    
    lead = (x_rep[:,:, 1:])
    lag = (x_rep[:,:, :-1])
    x_ll = torch.cat((lead, lag), dim=3)

    return x_ll

# Lead lag with time: input (windows, sequence length, feature dimension) output (windwos, lead_lag dim (= double feature dim -1), [time(lead),lead, lag])
def lead_lag_transform_with_time4(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    # gives back time interval between 0 adn 1 with the size of (batch/window, sequence length, 1)
    t = get_time_vector4(x.shape[0], x.shape[1],x.shape[2]).to(x.device)
    # we need 3 replicants of all the data for what we want to do
    t_rep = torch.repeat_interleave(t, repeats=3, dim=2)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=2)
    # concat the data, it will (almost) tripe in size
    x_ll = torch.cat([
        t_rep[:,:, 2:],
        x_rep[:,:, 1:-1],
        x_rep[:,:, :-2],
    ], dim=3)
    return x_ll
 
# Cat Lags
def cat_lags4(x: torch.Tensor, m: int) -> torch.Tensor:
    q = x.shape[2]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(q-m+1):
        x_lifted.append(x[:,:, i:i + m])
        x_lifted[-1] = x_lifted[-1].view(x_lifted[-1].shape[0],x_lifted[-1].shape[1], 1, x_lifted[-1].shape[2]*x_lifted[-1].shape[3])
    return torch.cat(x_lifted, dim=2).to(x.device)  

# Cumulative sum for input(x,y,z) out (x,y,v,b)
def cum_sum4(x: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim # what is the appropriate dim, depends if we use batch or not without its dim 0 with its dim 1
    return x.cumsum(dim=dim).to(x.device)
# stream preserving augmentation
############################## End of Augmentations 4 dim##############################


############################ start of Helpers ############################
def split_time_series(x: torch.Tensor, window: int) -> torch.Tensor:
    N = x.shape[0] // window
    result = torch.empty((N, window, x.shape[1]))

    for i in range(N):
        result[i, :, :] = x[i*window: window*(i+1),:]

    return result
############################ end of Helpers ############################

############################## Start of Multicolinearity dim ##############################
def calculate_vif(x):
    vif = [variance_inflation_factor(x.cpu().numpy(), i) for i in range(x.shape[1])]
    return torch.tensor(vif)

# Function to remove features with high VIF
def remove_vif(x, threshold=10):
    while True and x.shape[1] > 1:
        vif = calculate_vif(x)
        max_vif = torch.max(vif)  # check if largest value below threshold
        if max_vif <= threshold:
            break
        feature_to_remove = torch.argmax(vif)  # get index
        x = torch.cat((x[:, :feature_to_remove], x[:, feature_to_remove + 1:]), dim=1)  # remove highest vif
    return x
############################## End of Multicolinearity dim ##############################