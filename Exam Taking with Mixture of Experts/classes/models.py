import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=1, output_size=1, scale=1e-3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_size) * scale)
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size) * scale)
        """
        self.lr = .05
        self.W_hh = torch.ones(hidden_size, hidden_size)
        self.W_ih = torch.zeros(input_size, hidden_size)
        self.W_ih[0, 0] = -2*self.lr
        self.W_hi = torch.zeros(hidden_size, input_size)
        self.W_hi[0, 1] = -self.lr
        self.W_oh = torch.ones(output_size, hidden_size)
        """
        
    
    def forward(self, x):
        n, _, seq_len = x.size()
        h = torch.zeros(n, self.hidden_size)
        
        for t in range(seq_len):
            h = torch.einsum("hh,nh->nh", self.W_h, h) + (
                torch.einsum("ni,ih,nh->nh", x[:, :, t], self.W_ih, h) + 
                torch.einsum("hi,ni->nh", self.W_i, x[:, :, t]))
            o = h
        
        return o, h
    
    
class Router(nn.Module):
    def __init__(self, d_embedding, n_classes, scale=1e-3):
        super().__init__()
        
        self.W_out = nn.Parameter(torch.randn(n_classes, d_embedding) * scale)
        
    def forward(self, x):
        out = x
        out = torch.einsum("od,nd->no", self.W_out, x)
        return out    