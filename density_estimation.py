
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt

def gaussian (x, mean, variance):
    ''' Computes the probability of obtaining x under a gaussian distribution with
    given mean and variance.'''
    
    return 1 / np.sqrt(2*np.pi*variance) * np.exp (1/2*((x-mean)/variance)**2)

def generate_gaussian (mean=0, variance=1):
    ''' Samples 1000 times from a normal distribution.'''
    
    X = [np.random.normal (mean, np.sqrt(variance)) for time in range (1000)]
    return X
    
def estimate_gaussian (X):
    ''' Uses maximum likelihood to estimate a mean and variance under a gaussian distribution.'''
    
    sample_mean = sum(X) / len(X)
    sample_variance = sum([(x - sample_mean)**2 for x in X]) / len (X)
    
    return sample_mean, sample_variance

def uniform (x, start, end):
    ''' Computes the probability of obtaining x under a uniform distribution.'''
    
    return 1/(end-start)

def generate_uniform (start=0, end=1):
    ''' Samples 1000 times from a uniform distribution.'''
    
    X = [np.random.uniform (start, end) for time in range (1000)]
    return X

def estimate_uniform (X):
    ''' Uses maximum likelihood to compute a start and end under a uniform distribution.'''
    
    sample_start = min (X)
    sample_end = max (X)
    
    return sample_start, sample_end

def exponential (x, scale):
    ''' Computes the probability of obtaining x under an exponential distribution.'''
    
    if x < 0:
        return 0
    else:
        return scale*np.exp(-scale*x)

def generate_exponential (scale=1):
    ''' Samples 1000 times from an exponential distribution.'''
    
    X = [np.random.exponential (scale) for time in range (1000)]
    return X

def estimate_exponential (X):
    ''' Uses maximum likelihood to compute a scale under an exponential distribution.'''
    
    sample_scale = len (X) / sum (X)
    return sample_scale

def log_likelihood (distribution, X):
    ''' Computes the log likelihood given a distribution and a set of samples.'''
    
    total = 0
    for x in X:
        total += np.log (distribution (x))
    return total

def main():
    ''' Generates a data set under some distribution. Uses maximum likelihood to determine
    the distribution of best fit.'''
    
    X = generate_gaussian ()
    
    mean, variance = estimate_gaussian (X)
    start, end = estimate_uniform (X)
    scale = estimate_exponential (X)
    
    def g(x):
        return gaussian (x, mean, variance)
    def u(x):
        return uniform (x, start, end)
    def e(x):
        return exponential (x, scale)
    
    distributions = [g, u, e]
    
    max_function = None
    max_ll = -1000
    for distribution in distributions:
        if log_likelihood (distribution, X) > max_ll:
            max_function = distribution
    
    if max_function == g:
        print ('Gaussian:', mean, variance)
    if max_function == u:
        print ('Uniform:', start, end)
    if max_function == e:
        print ('Exponential:', scale)
    

main()

