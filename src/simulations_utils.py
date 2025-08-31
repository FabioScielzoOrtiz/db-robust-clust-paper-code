########################################################################################################################################################################

import pandas as pd
import polars as pl
import numpy as np
from sklearn.datasets import make_blobs
from FastKmedoids.data import outlier_contamination


########################################################################################################################################################################

def process_simulated_data(X):
    
    # numpy array to pandas df
    X = pd.DataFrame(X)     

    # Column names 
    X.columns = [f"X{i}" for i in range(1, X.shape[1]+1)]

    # quant variables converted to binary and multi-class
    X['X5'] = pd.cut(X['X5'], bins=[X['X5'].min()-1, X['X5'].mean(), X['X5'].max()+1], labels=False)
    X['X6'] = pd.cut(X['X6'], bins=[X['X6'].min()-1, X['X6'].mean(), X['X6'].max()+1], labels=False)
    X['X7'] = pd.cut(X['X7'], bins=[X['X7'].min()-1, X['X7'].quantile(0.25), X['X7'].quantile(0.50), X['X7'].quantile(0.75), X['X7'].max()+1], labels=False)
    X['X8'] = pd.cut(X['X8'], bins=[X['X8'].min()-1, X['X8'].quantile(0.25), X['X8'].quantile(0.50), X['X8'].quantile(0.75), X['X8'].max()+1], labels=False)   

    return X

########################################################################################################################################################################

def get_simulation_1(random_state=123, n_samples=35000, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=[2,2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=123)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=123)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate(outliers_idx_X1, outliers_idx_X2))
        return X, y, outliers_idx
    else:
        return X, y

########################################################################################################################################################################




########################################################################################################################################################################

def get_simulation_3(random_state=123, n_samples=100000, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=[2,2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=123)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=123)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate(outliers_idx_X1, outliers_idx_X2))
        return X, y, outliers_idx
    else:
        return X, y

########################################################################################################################################################################

def get_simulation_4(random_state=123, n_samples=300000, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=[2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=123)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=123)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate(outliers_idx_X1, outliers_idx_X2))
        return X, y, outliers_idx
    else:
        return X, y
    
########################################################################################################################################################################

def get_simulation_7(random_state=123, n_samples=300000):
        
    # Data simulation
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=[2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    return X, y
    
########################################################################################################################################################################