import numpy as np
def scale_data_std(X):
    return (X-np.mean(X, axis=0))/np.std(X, axis=0)
def scale_data_minmax(X):
    return (X-np.min(X, axis=0))/np.max(X, axis=0)