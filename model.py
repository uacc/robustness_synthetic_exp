import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
from collections import defaultdict

class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_layer_number, hidden_layer_list):
        # hidden_layer_number: number of hidden layer in the network
        # hidden_layer_list: width of each hidden layer in the network. len(hidden_layer_list) should be equal to hidden_layer_number
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_layer_number  = hidden_layer_number
        self.hidden_layer_list = hidden_layer_list
        if len(self.hidden_layer_list) != self.hidden_layer_number:
            print("Wrong network setup!")
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_layer_list[0])
            self.relu1 = torch.nn.ReLU()
            for i in range(self.hidden_layer_number):
                attr_name = 'fc' + str(i + 2)
                if i + 1 < self.hidden_layer_number:
                    setattr(self, attr_name, torch.nn.Linear(self.hidden_layer_list[i], self.hidden_layer_list[i+1]))
                    act_name = 'relu' + str(i + 2)
                    setattr(self, act_name, torch.nn.ReLU())
                else:
                    setattr(self, attr_name, torch.nn.Linear(self.hidden_layer_list[i], 1))
                    act_name = 'sigmoid'
                    setattr(self, act_name, torch.nn.Sigmoid())
        
    def forward(self, x):
        for i in range(self.hidden_layer_number + 1):
            attr_name = 'fc' + str(i + 1)
            layer = getattr(self, attr_name)
            x = layer(x)
            if i + 1 < self.hidden_layer_number:
                act_name = 'relu' + str(i+1)
                act = getattr(self, act_name)
                x = act(x)
            else:
                act_name = 'sigmoid'
                act = getattr(self, act_name)
                x = act(x)
        return x
