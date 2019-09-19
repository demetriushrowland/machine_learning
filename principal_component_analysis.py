
# coding: utf-8

# In[13]:


import numpy as np
import numpy.linalg as la

def singular_value_decomposition (matrix):
    '''Computes the singular value decomposition of a D x N matrix. Returns the
    matrices U, S, V such that matrix = U @ S @ V.T'''
    SVD = la.svd (matrix)
    U = SVD[0]
    S = np.diag(SVD[1])
    V = SVD[2].T
    return U, S, V

def get_principal_components (data, d):
    '''Computes the d-dimensional affine subspace of best fit for the points in data.
    Returns the matrix U whose columns form a basis for this subspace and the matrix Y
    whose columns consist of the projections of each point in data onto the subspace.'''
    N, D = data.shape
    X = data.T
    mu = np.mean (X, axis=1)
    X = X - np.array([list(mu),]*N).T
    U, S, V = singular_value_decomposition (X)
    U = U[:, 0:d]
    Y = (S@V.T)[0:d, :]
    return U, Y

def main():
    '''Generates a 4 x 100 matrix consisting of the same point in R^100 repeated 4 times.'''
    data = np.array([[1,2,-1,1]])
    for time in range (99):
        x = np.array(data[0])
        x = list(x)
        x = np.array([x])
        data = np.concatenate ((data, x), axis=0)
    data = data.T
    N, D = data.shape
    mu = np.mean(data, axis=0)
    mu_matrix = np.array([list(mu),]*N).T
    # Finds the principal components of data for reduced dimension d = 1
    U,Y = get_principal_components (data,1)
    # Computes the coordinates for the projection of each point in data (given by Y) in R^100
    projections = mu_matrix + U@Y
    return projections

main()
        

