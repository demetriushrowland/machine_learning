import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import expon
from scipy.stats import t
from scipy.stats import nbinom
directory = '/Users/Zhonghou/Desktop/General/Classes/Statistical Modeling I/Project/'

class MH_block_sampler:
    def __init__(self, X, d, p, g, prior):
        self.X = X
        self.d = d
        self.p = p
        self.n = len(X)
        self.g = g
        self.thetas = [0]
        self.prior = prior

    def q (self, old_theta):
        theta = norm.rvs(loc=old_theta)
        return theta

    def find_lam (self, theta):
        def f(lam):
            total = 0
            for i in range(self.n):
                total = np.add(total, np.exp (np.dot(lam, self.g(self.X[i], theta))), casting='unsafe')
            return total
        
        return minimize(f, x0=[1 for i in range(self.d)]).x

    def find_p (self, theta):
        probabilities = []
        lam = self.find_lam (theta)
        for i in range(self.n):
            probabilities.append (np.exp(np.dot(lam, self.g(self.X[i], theta))))
        sum_prob = sum(probabilities)
        probabilities = [p / sum_prob for p in probabilities]
        return probabilities

    def find_alpha (self, theta):
        numerator = norm.pdf (self.thetas[-1], loc=self.thetas[-1])
        denominator = norm.pdf (theta, loc = self.thetas[-1])
        C = numerator / (denominator)
        new_probabilities = self.find_p (theta)
        log_new_probabilities = [np.log (p) for p in new_probabilities]
        product = np.exp(np.sum(log_new_probabilities) + 7.5*self.n)
        numerator = product*1e20 * self.prior (theta)
        old_probabilities = self.find_p (self.thetas[-1])
        log_old_probabilities = [np.log (p) for p in old_probabilities]
        product = np.exp(np.sum(log_old_probabilities) + 7.5*self.n)
        denominator = product*1e20 * self.prior(self.thetas[-1])
        alpha = min([1, C*numerator/(denominator)])
        return alpha

    def find_u (self):
        return np.random.uniform(0, 1)

    def sample (self, num_samples):
        for t in range(num_samples):
            theta = self.q(self.thetas[-1])
            alpha = self.find_alpha(theta)
            u = self.find_u()
            if u <= alpha:
                self.thetas.append (theta)
            else:
                self.thetas.append (self.thetas[-1])

    def burn_and_thin (self):
        S = len(self.thetas)
        burned_thetas = []
        for s in range(S):
            if s >= 1000:
                if s % 3 == 0:
                    burned_thetas.append(self.thetas[s])
        self.thetas = burned_thetas

def toy():
    ''' Model: x = c + epsilon, with c unknown, and epsilon ~ Norm(0, 1)'''
    X = [0 + norm.rvs() for time in range(100)]
    def g(x, theta):
        return [x - theta]

    def prior(theta):
        return norm.pdf (theta, loc=-3)

    mh_block_sampler = MH_block_sampler (X, 1, 1, g, prior)

    mh_block_sampler.sample(10000)
    mh_block_sampler.burn_and_thin()
    plt.hist([mh_block_sampler.thetas[i] for i in range(len(mh_block_sampler.thetas))], density=True, bins=30)
    plt.show()

    with open (directory+'thetas.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Theta'])
        for theta in mh_block_sampler.thetas:
            writer.writerow([theta])
    
    return

def nuclear_test ():
    with open (directory+'nuclear_data.txt') as text_file:
        reader = text_file.read()
        reader = reader.split('\n')
        i = 0
        for i in range(len(reader)):
            reader[i] = reader[i].split('\t')
    nuclear_data = pd.DataFrame(reader)
    radii = []
    energies = []
    for i in range(942):
        energies.append(nuclear_data.iat[i,3])
        radii.append(nuclear_data.iat[i,14])
        try:
            energies[-1] = float(energies[-1])
            radii[-1] = float(radii[-1])
        except:
            energies.pop(-1)
            radii.pop(-1)

    X = [[radii[i], energies[i]] for i in range(len(radii))]

    def prior (x):
        return expon.pdf(x)


    def g(x, theta):
        return x[0] - np.exp(1/5*np.log(x[1])) * theta

    print(g([1, 2], 3))

    mh_block_sampler = MH_block_sampler (X, 1, 1, g, prior)
    mh_block_sampler.sample(10000)
    mh_block_sampler.burn_and_thin()
    #plt.hist([mh_block_sampler.thetas[i] for i in range(len(mh_block_sampler.thetas))], density=True, bins=30)
    #plt.show()
    probabilities = mh_block_sampler.find_p(10)
    marginal_likelihood_model1 = np.exp(np.sum([np.log(p) for p in probabilities])) * prior(10) / .026

    def g(x, theta):
        return x[0] - x[1]*theta

    mh_block_sampler = MH_block_sampler (X, 1, 1, g, prior)
    mh_block_sampler.sample(10000)
    mh_block_sampler.burn_and_thin()
    #plt.hist([mh_block_sampler.thetas[i] for i in range(len(mh_block_sampler.thetas))], density=True, bins=30
    #plt.show()
                                        
    probabilities = mh_block_sampler.find_p(85)
    marginal_likelihood_model2 = np.exp(np.sum([np.log(p) for p in probabilities])) * prior(85) / .013

    print(marginal_likelihood_model1, marginal_likelihood_model2)

def count_regression (n):
    X = [norm.rvs (.4, 1/9) for i in range(n)]
    U = [np.exp (1*x) for x in X]
    p = 1/2
    Y = [nbinom.rvs (p/(1-p)*u, p) for u in U]
    data = [[Y[i], X[i]] for i in range(n)]

    def g(x, theta):
        return (X[0] - np.exp(X[1]*theta))*X[1]

    def prior(theta):
        return t.pdf(x=theta, df=2.5, loc=0, scale=5)

    x_list = np.arange(-3, 3, .01)
    y_list = [prior(x) for x in x_list]
    plt.plot(x_list, y_list)
    plt.show()

    mh_block_sampler = MH_block_sampler (data, 1, 1, g, prior)
    mh_block_sampler.sample(1000)
    plt.hist([mh_block_sampler.thetas[i] for i in range(len(mh_block_sampler.thetas))], density=True, bins=30)
    plt.show()
    
    return

def iv_regression (n):
    W = [np.random.uniform (0, 1) for time in range(n)]
    Z = [norm.rvs(loc=.5, scale=1) for time in range(2*n)]
    Z1 = Z[:n]
    Z2 = Z[n:]
    EU = [multivariate_normal.rvs(mean=[0, 0], cov = [[1, .7], [.7, 1]]) for time in range(n)]
    E = [row[0] for row in EU]
    U = [row[1] for row in EU]
    X = [Z1[i] + Z2[i] + W[i] + U[i] for i in range(n)]
    Y = [1 + .5*X[i] + .7*W[i] + E[i] for i in range(n)]
    data = [[Y[i], X[i], W[i]] for i in range(n)]

    def g(X, theta):
        return X[0] - 1 - theta*X[1] - .7*X[2]

    def prior(theta):
        return t.pdf(x=theta, df=2.5, loc=0, scale=5)

    x_list = np.arange(-3, 3, .01)
    y_list = [prior(x) for x in x_list]
    plt.plot(x_list, y_list)
    plt.show()

    mh_block_sampler = MH_block_sampler (data, 1, 1, g, prior)
    mh_block_sampler.sample(6000)
    mh_block_sampler.burn_and_thin()
    plt.hist([mh_block_sampler.thetas[i] for i in range(len(mh_block_sampler.thetas))], density=True, bins=30)
    plt.show()
    return

    
count_regression(50)
