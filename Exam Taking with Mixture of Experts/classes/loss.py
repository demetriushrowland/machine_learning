import torch
import torch.nn as nn

class Loss(nn.Module):
    def forward(self, y_hat, y=None):
        error = torch.abs(y_hat - y)
        error = torch.min(error, dim=1).values
        n = error.size()[0]
        result = 1./n * torch.sum(error)
        return result