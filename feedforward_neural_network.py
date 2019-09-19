
# coding: utf-8

# In[23]:


import numpy as np
from scipy import optimize as op
import numpy.linalg as la

def feedforward (network, data):
    '''Runs the feature inputs through the network to generate estimates for the target values.'''
    
    inputs = data['inputs']
    activations = network['activations']
    weights = network['weights']
    num_points = inputs.shape[0]
    num_inputs, num_hidden, num_outputs = network['dimensions']
    estimates = np.zeros((num_points, num_outputs))
    
    total = 0
    for point_num in range (num_points):
        hidden_values = np.zeros(num_hidden)
        point = inputs[point_num]
        for hidden_num in range (num_hidden):
            weight = [weights[0][i][hidden_num] for i in range (num_inputs)]
            activation = activations[0][hidden_num]
            hidden_value = activation(np.dot(weight, point))
            hidden_values[hidden_num] = hidden_value
        
        for output_num in range (num_outputs):
            weight = [weights[1][i][output_num] for i in range (num_hidden)]
            activation = activations[1][output_num]
            output_value = activation(np.dot (weight, hidden_values))
            estimates[point_num][output_num] = output_value
    return estimates

def error_function (network, data):
    '''Generates estimates for the target values, and returns an error given by the 
    sum of squares error function.'''
    
    estimates = feedforward (network, data)
    outputs = data['outputs']
    error = .5*(la.norm (estimates - outputs))**2
    return error

def set_weights (network, weights):
    '''Provides a mechanism for fixing the network weights using a weight vector.'''
    
    num_inputs, num_hidden, num_outputs = network['dimensions']
    
    for input_num in range (num_inputs):
        for hidden_num in range (num_hidden):
            index = input_num*num_hidden + hidden_num
            network['weights'][0][input_num][hidden_num] = weights[index]
    
    for hidden_num in range (num_hidden):
        for output_num in range (num_outputs):
            index = num_inputs*num_hidden + hidden_num*num_outputs + output_num
            network['weights'][1][hidden_num][output_num] = weights[index]
            

def main():
    ''' Generates features and target values by sampling from two uniform distributions on [-5,5]
    (say, for random variables x0 and x1) and setting y = 5*(x0-x1)**2 + 2*(x0+3*x1+1)**3 + 10.
    Feeds this data into the network to generate an error function in the space of weights.
    Uses basinhopping to minimize the error function to return appropriate weight values.
    Prints the final estimate for each set of features and the corresponding target value.'''
    
    
    def f(x):
        return x**3
    
    def g(x):
        return x**2
    
    def h(x):
        return x
    
    activations = [[f, g, h], [h]]
    inputs = np.array([[np.random.uniform (-5,5), np.random.uniform (-5, 5), 1] for time in range (100)])
    outputs = np.array([[5*(x[0]-x[1])**2 + 2*(x[0]+3*x[1]+x[2])**3 + 10*x[2]] for x in inputs])
    weights = np.array([[[0,0], [0,0], [0,0]], [[0],[0]]])
    dimensions = (3,2,1)
    network = {'weights': weights, 'activations': activations, 'dimensions': dimensions}
    data = {'inputs': inputs, 'outputs': outputs}
    num_points = inputs.shape[0]
    
    def error (weights):
        set_weights (network, weights)
        return error_function (network, data)
    
    weights = op.basinhopping (error, [0,1,1,0,1,0,2,5]).x
    set_weights (network, weights)
    
    #set_weights (network, [0,1,1,0,2,5])
    
    estimates = feedforward (network, data)
    
    for point_num in range (num_points):
        print ('Input:', inputs[point_num])
        print ('Output:', outputs[point_num])
        print ('Estimates:', estimates[point_num])
        print ('')
    
    return network['weights']

main()

