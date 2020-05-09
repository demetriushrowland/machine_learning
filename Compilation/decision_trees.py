import numpy as np
from scipy.optimize import minimize_scalar

class Tree:
    def __init__(self, num_nodes, data):
        self.data = data
        self.shape = data.shape
        self.N = self.shape[0]
        self.p = self.shape[1] - 1
        self.X = self.data[:, :self.p-1]
        self.y = self.data[:, -1]
        self.num_nodes = num_nodes
        self.j_array = np.array([])
        self.s_array = np.array([])

    def split(self):
        
        for j in range(self.p):
            def f(s):
                y1 = self.y[self.X[:, j] <= s]
                y2 = self.y[self.X[:, j] > s]
                c1 = np.mean(y1)
                c2 = np.mean(y2)
                return np.sum((y1-c1)**2) + np.sum((y2-c2)**2)

            result = minimize_scalar(f)
            s = result.x
            error = result.fun
            if j == 0:
                min_error = error
                min_j = j
                min_s = s
            else:
                if error < min_error:
                    min_j = j
                    min_s = s
                    
        return min_j, min_s

    
            
            
            
                
                
                    
                
                
            

def main():
    return
