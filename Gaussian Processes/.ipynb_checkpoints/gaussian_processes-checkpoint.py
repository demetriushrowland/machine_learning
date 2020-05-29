import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
import time
from scipy.stats import norm
from scipy.stats import multivariate_normal as MVN
from scipy.stats import uniform
from scipy.special import expit
from scipy.special import softmax
from scipy.integrate import quad
from scipy.optimize import basinhopping

class GP:
    def __init__ (self, compute_mean = lambda x: 0,
                  compute_variance = lambda x, y: np.exp(-(1/2)*la.norm((x - y))**2)):
        self.compute_mean = compute_mean
        self.compute_variance = compute_variance

    def compute_mean_vector (self, X):
        n = X.shape[0]
        mean = np.zeros(n)
        for i in range(n):
            mean[i] = self.compute_mean(X[i])
        return mean

    def compute_covariance_matrix (self, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        covariance = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                covariance[i][j] = self.compute_variance(X1[i], X2[j])
        return covariance

    def sample_prior (self, X):
        n = X.shape[0]
        mean = self.compute_mean_vector (X)
        variance = self.compute_covariance_matrix (X, X)
        samples = MVN.rvs(mean, variance)
        return samples

    def regress (self, X_train, Y_train, X_test):
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        cov_X_test_train = self.compute_covariance_matrix (X_test, X_train)
        cov_X_train_train = self.compute_covariance_matrix (X_train, X_train)
        cov_X_train_test = self.compute_covariance_matrix (X_train, X_test)
        cov_X_test_test = self.compute_covariance_matrix (X_test, X_test)

        epsilon = 10e-6
        cov_X_train_train += epsilon*np.identity(n_train)
        L = la.cholesky(cov_X_train_train, lower=True)
        L_inv = la.solve_triangular(L, np.identity(n_train), lower=True)
        cov_X_train_train_inv = np.matmul(L_inv.T, L_inv)

        posterior_mean = np.matmul(cov_X_test_train, np.matmul(cov_X_train_train_inv,
                            np.reshape(Y_train, (n_train, 1))))
        posterior_mean = posterior_mean.flatten()
        posterior_variance = cov_X_test_test - np.matmul(cov_X_test_train,
                            np.matmul(cov_X_train_train_inv, cov_X_train_test))
                
        samples = MVN.rvs(posterior_mean, posterior_variance)
        return samples

    def classify (self, X_train, Y_train, X_test, num_classes=2):
        if num_classes == 2:
            samples = self.regress(X_train, Y_train, X_test)
            n_test = X_test.shape[0]
            pi_test = np.zeros(n_test)
            Y_estimates = np.zeros(n_test)
            for i in range(n_test):
                p = samples[i]
                pi_test[i] = p
                if p < .5:
                    Y_estimates[i] = 0
                else:
                    Y_estimates[i] = 1

            return pi_test, Y_estimates           

        else:     
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]
            C = num_classes
            I = np.identity(C)
            pi_mat = np.zeros((n_test, C))
            for c in range(C):
                X_train_c = []
                Y_train_c = []
                class_label = I[c]
                for n in range(n_train):
                    if np.array_equal(Y_train[n], class_label):
                        X_train_c.append(list(X_train[n]))
                        Y_train_c.append(1)
                    else:
                        X_train_c.append(list(X_train[n]))
                        Y_train_c.append(0)
                X_train_c = np.array(X_train_c)
                Y_train_c = np.array(Y_train_c)
                pi_test_c, Y_estimates_c = self.classify(X_train_c, Y_train_c, X_test)
                pi_mat[:, c]= pi_test_c

            Y_estimates = np.zeros((n_test, C))
            max_indices = np.argmax(pi_mat, axis=1)
            for i in range(n_test):
                Y_estimates[i] = I[max_indices[i]]
            return Y_estimates
            
    def get_error(Y_estimates, Y_test, option):
        n_test = Y_test.shape[0]
        if option == "regress":
            error = np.sqrt(1/n_test)*la.norm(Y_test - Y_estimates)
            return error
        if option == "classify":
            errors = np.zeros(n_test)
            for i in range(n_test):
                if len(Y_test.shape) == 1:
                    errors[i] = float(not np.equal(Y_test[i], Y_estimates[i]))
                else:
                    errors[i] = float(not np.array_equal(Y_test[i], Y_estimates[i]))
            error = la.norm(errors)**2 / n_test
            return error
                            
                    
                    
                
              
                    
            
                
                




