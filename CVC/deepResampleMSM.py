import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.autograd import Variable, backward

########################################################################
#                                                                      # 
#                                                                      #
#                                                                      #
#                                                                      #
#                                                                      #
#                                                                      #
########################################################################

class Prinz:
    def __init__(self, x0, kT, D, dt):
        self.x0 = x0
        self.kT = kT
        self.D = D
        self.dt = dt
        
    def potential_function(x):
        return 4*(x**8+0.8*np.exp(-80*x*x)+0.2*np.exp(-80*(x-0.5)**2)+0.5*np.exp(-40*(x+0.5)**2))

    def potential_derivative(x):
        return 32*(x**7 - 16*x*np.exp(-80*x*x) - 4*(x-.5)*np.exp(-80*(x-.5)**2) - 5*(x+.5)*np.exp(-40*(x+.5)**2))

    def generate_data (self, T):
        data = np.zeros(T)
        data[0] = self.x0
        coeff = np.sqrt(2*self.dt*self.D)
        for t in range(T-1):
            x_t = data[t]
            e_t = norm.rvs()
            data[t+1] = x_t - self.dt/self.kT*Prinz.potential_derivative(x_t) + coeff*e_t
        return data

class Chi(nn.Module):
    def __init__(self, num_input, num_hidden_layers, num_hidden, num_output):
        super(Chi, self).__init__()
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_hidden_layers = num_hidden_layers
        self.num_output = num_output
        self.input_linear = nn.Linear(num_input, num_hidden)
        self.hidden_linear = nn.Linear(num_hidden, num_hidden)
        self.output_linear = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.input_linear(x)
        x = F.relu (x)
        for hidden_layer in range(self.num_hidden_layers):
            x = self.hidden_linear(x)
            x = F.relu(x)
        x = self.output_linear(x)
        x = F.softmax(x)
        return x

class Gamma(nn.Module):
    def __init__(self, num_input, num_hidden_layers, num_hidden, num_output):
        super(Gamma, self).__init__()
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_hidden_layers = num_hidden_layers
        self.num_output = num_output
        self.input_linear = nn.Linear(num_input, num_hidden)
        self.hidden_linear = nn.Linear(num_hidden, num_hidden)
        self.output_linear = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.input_linear(x)
        x = F.relu(x)
        for hidden_layer in range(self.num_hidden_layers):
            x = self.hidden_linear(x)
            x = F.relu(x)
        x = self.output_linear(x)
        x = F.relu(x)
        return x

def train(data, chi, gamma, tau):
    optimizer = torch.optim.Adam(list(chi.parameters())+list(gamma.parameters()))
    batch_size = 100
    
    X_train = torch.from_numpy(data[:-tau]).float()
    X_train = torch.reshape(X_train, (995, 1))
    Y_train = torch.from_numpy(data[tau:]).float()
    Y_train = torch.reshape(Y_train, (995, 1))
    data_size = 995

    for epoch in range(200):
        total_error = 0
        index_set = torch.randperm(data_size)
        count = 0
        while True:
            actual_batch_size = min(batch_size, data_size - count)
            if actual_batch_size <= 0:
                break
            X_input = Variable(X_train[index_set[count:count+actual_batch_size]])
            Y_input = Variable(Y_train[index_set[count:count+actual_batch_size]])
            count += actual_batch_size
            chi_val = chi(X_input)
            gamma_val = gamma(Y_input)
            gamma_array = gamma(Y_train)
            gamma_norm = 1/data_size*np.sum(gamma_array.detach().numpy(), axis=0)
            gamma_normed = torch.tensor(np.divide(gamma_val.detach().numpy(), gamma_norm))
            output = torch.mm(gamma_normed, torch.t(chi_val))
            loss = -torch.sum(torch.log(output))
            optimizer.zero_grad()
            backward(loss)
            optimizer.step()
            
        
    return

def compute_transition_matrix(data, chi, gamma, tau):
    data_lagged = data[t

def main():
    tau = 5
    prinz = Prinz(-.75, 1.45, 1, .01)
    data = prinz.generate_data(1000)
    chi = Chi(1, 4, 64, 4)
    gamma = Gamma(1, 4, 64, 4)
    train(data, chi, gamma, tau)

main()
