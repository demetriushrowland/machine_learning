
# coding: utf-8

# In[25]:


import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt

def linear_regression (X, y):
    ''' Given a set of predictor values X and target values y,
    computes the vector of weights that determines a linear model of best fit.'''
    
    weights = np.matmul (np.matmul (la.inv (np.matmul (X.T, X)), X.T), y)
    return weights

def main():
    ''' Performs linear regression on a simulated data set with practice weights 
    [5, 6]. Plots the resulting line of best fit.'''
    
    X = []
    y = []
    practice_weights = [5, 6]
    for time in range (100):
        X.append ([1, np.random.uniform(-10,10)])
        y.append (np.dot(X[-1], practice_weights) + np.random.normal (0, 5))
    
    # Computes the weights for the linear model that best fits X, y
    X = np.array(X)
    y = np.array(y)
    weights = linear_regression (X, y)

    # Computes y estimates given the computed weights
    x_test = [x/100 - 10 for x in range (2000)]
    y_test = [np.dot (np.array([1,x]), weights) for x in x_test]
    x = X.T[1]
    
    plt.scatter (x_test, y_test)
    plt.scatter (x, y)
    
    plt.show()
    
    print ('Weights:', weights[0], weights[1])

main()
    

