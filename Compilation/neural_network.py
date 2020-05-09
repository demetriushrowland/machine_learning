import numpy as np

class Net:
    def __init__ (self, weights, activations):
        self.weights = weights
        self.num_input = weights[0].shape[1]
        self.num_hidden_layers = len(weights)-1
        self.num_hidden = [weights[i].shape[1] for i in range(1, len(weights))]
        self.num_output = weights[-1].shape[0]
        self.activations = activations

    def forward (self, x):
        weights = self.weights
        activations = self.activations
        x = np.matmul(weights[0], x)
        x = activations[0](x)
        for hidden_layer_num in range(self.num_hidden_layers):
            x = np.matmul(weights[hidden_layer_num+1], x)
            x = activations[hidden_layer_num+1](x)
        return x

def main():
    return
        
