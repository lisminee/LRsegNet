
import torch

from utils.utils import *


import torch.nn as nn

def normalization(data):
    _range = np.max(abs(data))
    return data / _range


def pdist_torch(input, eps=1e-5):
    x = input.clone()
    n = x.size(0)
    subtracted = x.view(n, 1, -1) - x.view(1, n, -1)
    dist = torch.sqrt((subtracted**2).sum(dim=-1)+eps)
    return dist

def normalization_torch(data):
    _range = torch.max(torch.abs(data))
    return data / _range


def contrastive_graph_class_similarity_loss(x_flt, y_pred, alpha=1e-3, eps=1e-5):
  

    diff_x = pdist_torch(x_flt, eps)
    diff_y = pdist_torch(y_pred, eps)
    

    diff_x = normalization_torch(diff_x)
    diff_y = normalization_torch(diff_y)
    

    diff_x = torch.triu(diff_x)
    diff_y = torch.triu(diff_y)
    

    class_loss = torch.sum((diff_x - diff_y) ** 2)
    

    regularization_loss = torch.sum(torch.abs(y_pred))
    

    total_loss = class_loss + alpha * regularization_loss
    
    return total_loss / len(x_flt)



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.fc4 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc5 = nn.Linear(2 * hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        return x

