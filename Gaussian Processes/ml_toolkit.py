import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCA:
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data, axis=0)
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        
    def run_svd(self):
        data = self.data
        N = self.N
        D = self.D
        K = min(N, D)
        X = np.zeros((D, N))
        for j in range(N):
            X[:, j] = data.T[:, j] - self.mean
        U_X, S_X, Vh_X = la.svd(X, full_matrices=False)
        self.U_X = U_X
        self.S_X = S_X
        self.Vh_X = Vh_X
    
    def run_pca(self, d):
        self.d = d
        N = self.N
        D = self.D
        K = min(N, D)
        U_X = self.U_X
        S_X = self.S_X
        Vh_X = self.Vh_X
        r = S_X.shape[0]
        U = U_X[:, :d]
        S = np.zeros((K, K))
        S[:r, :r] = np.diag(S_X)
        Y = (S @ Vh_X)[:d, :]
        
        self.U = U
        self.Y = Y
        
    def get_principal_components(self):
        return self.U
    
    def get_y(self):
        return self.Y
    
    def get_projections(self):
        proj = np.zeros((self.D, self.N))
        for j in range(self.N):
            proj[:, j] = self.mean + self.U[:, :self.d] @ self.Y[:self.d, j]
        return proj
    
class Net(nn.Module):
    def __init__(self, num_input, num_hidden, num_hidden_layers, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_hidden_layers = num_hidden_layers
        self.num_output = num_output
        self.input_layer = nn.Linear(num_input, num_hidden)
        self.hidden_layer = nn.Linear(num_hidden, num_hidden)
        self.output_layer = nn.Linear(num_hidden, num_output)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for hidden_layer_num in range(self.num_hidden_layers):
            x = self.hidden_layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x
    

class Linear_Model:
    def __init__(self):
        pass
    
    def classify(self, X_train, Y_train):
        n_train, n_features = X_train.shape
        X_train_with_intercept = np.ones((n_train, n_features + 1))
        X_train_with_intercept[:, :n_features] = X_train
        for j in range(n_train):
            x = X_train_with_intercept[j]
            
        pass
    
    def regress(self):
        pass
    
    