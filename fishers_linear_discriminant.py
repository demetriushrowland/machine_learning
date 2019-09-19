
# coding: utf-8

# In[11]:


import numpy as np
import numpy.linalg as la

def get_discriminant_data (group0, group1):
    n = group0.shape[1]
    within_class_covariance = np.zeros((n, n))
    N0 = group0.shape[0]
    N1 = group1.shape[0]
    
    mean0 = np.zeros (n)
    for x in group0:
        x = np.array (x)
        mean0 += x
    mean0 = 1/N0 * mean0
    
    mean1 = np.zeros (n)
    for x in group1:
        x = np.array (x)
        mean1 += x
    mean1 = 1/N1 * mean1
    
    for x in group0:
        x = np.array (x)
        within_class_covariance += np.matmul (x - mean0, (x - mean0).T)
    
    for x in group1:
        x = np.array (x)
        within_class_covariance += np.matmul (x - mean1, (x - mean1).T)
    
    return within_class_covariance, mean0, mean1

def get_weights (group0, group1):
    within_class_covariance, mean0, mean1 = get_discriminant_data (group0, group1)
    return np.matmul (la.inv (within_class_covariance), mean1 - mean0)

def main():
    group = np.array([[np.random.normal (), np.random.normal()] for time in range (1000)])
    practice_weights = np.array([1,1])
    
    group0 = []
    group1 = []
    for point in group:
        if np.dot(practice_weights, point) >= 0:
            group0.append (point)
        else:
            group1.append (point)
    group0 = np.array(group0)
    group1 = np.array(group1)
    
    return get_discriminant_data (group0, group1)
    
    #weights = get_weights (group0, group1)
    #return weights

main()

