import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd.variable import Variable

class QNetwork(nn.Module):
    def __init__(self,in_channel=6,nb_hidden=64,action_size=3,seed=1412):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.feature = nn.Sequential(
            nn.Conv1d(in_channel,12,kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv1d(12, 24, kernel_size=6, stride=3),
            nn.ReLU()
        )
        self.feature_size = 192
        self.head = nn.Sequential(
            nn.Linear(self.feature_size, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, action_size)
        )
        
    def forward(self, state):
        x = self.feature(state)
        x = x.view(-1,self.feature_size)
        x = self.head(x)
        return(x)