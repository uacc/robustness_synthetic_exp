import numpy as np
import os
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

def acc(net, data, targets):
    err, total = 0, 0
    for i, j in zip(data, targets):
        output = net(i)
        output = output.cpu().detach().numpy()
        if j != (np.sign(output[0] - 0.5) + 1) /2:
            err += 1
        total += 1
    return (total - err) / total

def test_loss(data, targets, D, org_net, lr):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(org_net.parameters(), lr = lr)
    org_net.eval()
    y_pred = org_net(data)
    before_train = criterion(y_pred.squeeze(), targets)
    
    return before_train.item()

def adv_train_set(data, attack_data, targets, attack_targets):
    merge_data = torch.cat((data, attack_data), 0)
    merge_targets = torch.cat((targets, attack_targets), 0)
    return merge_data, merge_targets

def gen_x_2d_range(N, D, R):
    res = []
    for i in range(N):
        u = np.random.normal(0,1,D)
        denom = (np.sum(u**2))**0.5
        ran = R * np.random.random()
        res.append(ran*u/denom)
    res = np.array(res)
    return res

def get_label(outputs):
    y = []
    for output in outputs:
        output = output.cpu().detach().numpy()
        y.append((np.sign(output[0] - 0.5) + 1) /2)
    return y

def get_label_tensor(outputs):
    res = (torch.sign(outputs - 0.5) + 1)/2
    return res

def get_x_0_1_tensor(x, y):
    x_1 = [data.cpu().detach().numpy() for i, data in enumerate(x) if y[i] > 0]
    x_1 = np.array(x_1)
    x_0 = [data.cpu().detach().numpy() for i, data in enumerate(x) if y[i] <= 0.]
    x_0 = np.array(x_0)
    return x_1, x_0

def get_x_0_1(x, y):
    x_1 = [data for i, data in enumerate(x) if y[i] > 0]
    x_1 = np.array(x_1)
    x_0 = [data for i, data in enumerate(x) if y[i] <= 0.]
    x_0 = np.array(x_0)
    return x_1, x_0


def get_mean(risk_dict, epsilons):
    mean = []
    for eps in epsilons:
        tmp = []
        for risk in risk_dict[str(eps)]:
            tmp.append(risk[1])
        tmp = np.array(tmp)
        mean.append(np.mean(tmp))
    mean = np.array(mean)
    return mean

def get_std(risk_dict, epsilons):
    std = []
    for eps in epsilons:
        tmp = []
        for risk in risk_dict[str(eps)]:
            tmp.append(risk[1])
        tmp = np.array(tmp)
        std.append(np.std(tmp))
    std = np.array(std)
    return std


def get_mean_meas(risk_dict, epsilons):
    mean = []
    for eps in epsilons:
        tmp = []
        for risk in risk_dict[str(eps)]:
            tmp.append(risk[1]+risk[0])
        tmp = np.array(tmp)
        mean.append(np.mean(tmp))
    mean = np.array(mean)
    return mean

def get_std_meas(risk_dict, epsilons):
    std = []
    for eps in epsilons:
        tmp = []
        for risk in risk_dict[str(eps)]:
            tmp.append(risk[1]+risk[0])
        tmp = np.array(tmp)
        std.append(np.std(tmp))
    std = np.array(std)
    return std    
def get_mean_adv(risk_dict, epsilons):
    mean = []
    for eps in epsilons:
        tmp = []
        for risk in risk_dict[str(eps)]:
            tmp.append(risk[2])
        tmp = np.array(tmp)
        mean.append(np.mean(tmp))
    mean = np.array(mean)
    return mean

def get_std_adv(risk_dict, epsilons):
    std = []
    for eps in epsilons:
        tmp = []
        for risk in risk_dict[str(eps)]:
            tmp.append(risk[2])
        tmp = np.array(tmp)
        std.append(np.std(tmp))
    std = np.array(std)
    return std   
