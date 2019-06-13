import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Function
from torch.nn.parameter import Parameter


class TraditionalLayer(nn.Module):

    def __init__(self, d_in, d_out, dropout=.5):
        super().__init__()

        self.linear = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        inputs:
            x.shape = batch_size x d_in
        return:
            y.shape = batch_size x d_out
        '''
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
