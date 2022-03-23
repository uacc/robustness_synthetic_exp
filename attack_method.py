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
from utils import *
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
from collections import defaultdict

class PGDAttack:
    # setup model and pgd attack parameters
    def __init__(self, model, eps, alpha, iters):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        
    def pgd_attack(self, images, labels) :
        # LinPGD Attack 
        loss = nn.BCELoss()
        ori_images = images
        device = torch.device("cuda" if True else "cpu")

        for i in range(self.iters) :   
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs.squeeze(), labels).to(device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            ori_image_min = torch.min(images).data
            ori_image_max = torch.max(images).data
            images = torch.clamp(ori_images + eta, min = ori_image_min, max = ori_image_max).detach_()

        return images
    
    
class InManiAttack:
    def __init__(self, model, eps, alpha, method, stepsize, iters):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.method = method
        self.stepsize = stepsize
        self.iters = iters
        
    def inmani_attack(self, images, labels):
        # in-manifold attack method
        if self.method == 1:
            return self._method1_attack(images, labels)
        else:
            return self._method2_attack(images, labels)
        
    def unit_proj(self, images):
        # project out-of-manifold data back to unit circle or xy-plane
        N = images.shape[0]
        D = images.shape[1]
        res = torch.ones(N, D)
        if D == 2:
            for i, d in enumerate(images):
                res[i] = torch.div(d, torch.norm(d, p = 2))
        else:
            for i, d in enumerate(images):
                res[i][0:2] = d[0:2]
                res[i][2] = 0.0
        return res.to('cuda')
        
    def _method1_attack(self, images, labels):
        # generate pgd attack data and project back to original manifold 
        pgd = PGDAttack(self.model, self.eps, self.alpha, self.iters)
        pgd_attack_images = pgd.pgd_attack(images, labels)
        in_manifold_images = self.unit_proj(pgd_attack_images)
        return in_manifold_images
        
    def _method2_attack(self, images, labels):
        # grid search method
        if images.shape[1] == 2:
            return self._method22d_attack(images, labels)
        else:
            return self._method23d_attack(images, labels)
        
    def _method22d_attack(self, images, labels):
        self.model.to('cuda')
        search_range = [-1 * self.eps + i * 2 * self.eps/self.stepsize for i in range(self.stepsize)]
        res = []
        loss = nn.BCELoss()
        for data, target in zip(images, labels):
            max_diff = 0
            done_pert = False
            tmp = data
            for pert in search_range:
                for pert_1, pert_2 in [(pert, pert), (pert, -1 * pert), (-1 * pert, pert), (-1 * pert, -1 * pert)]:
                    adv = torch.Tensor(2)
                    adv[0] = data[0] + pert_1
                    adv[1] = data[1] + pert_2
                    adv = torch.div(adv, torch.norm(adv, p=2)).to('cuda')
                    outputs = self.model(adv)
                    cost = loss(outputs.squeeze(), target).to(device)
                   if cost >= max_diff:
                        tmp = adv
                        max_diff = cost
                if done_pert:
                    break
            if not done_pert:
                res.append(tmp.cpu().detach().numpy())                
        res = np.array(res)
        res = torch.from_numpy(res).to('cuda')
        return res
    
    def _method23d_attack(self, images, labels):
        res = []
        search_range = [-1 * self.eps + i * 2 * self.eps/self.stepsize for i in range(self.stepsize)]
        loss = nn.BCELoss()
        for data, target in zip(images, labels):
            max_diff = 0
            done_pert = False
            tmp = data
            for pert_1 in search_range:
                for pert_2 in search_range:
                    adv = torch.Tensor(3)
                    adv[0] = data[0] + pert_1
                    adv[1] = data[1] + pert_2
                    adv[2] = 0
                    adv = adv.to('cuda')
#                     loss = self.model(adv)
#                     label = (torch.sign(loss - 0.5) + 1) / 2 
                    outputs = self.model(adv)
                    cost = loss(outputs.squeeze(), target).to(device)
#                     if label != target:
#                         res.append(adv.cpu().detach().numpy())
#                         done_pert = True
#                         break
#                     else:
#                         if torch.abs(loss - target) > max_diff:
#                             tmp = adv
#                             max_diff = torch.abs(loss - target)
                    if cost >= max_diff:
                        tmp = adv
                        max_diff = cost
                if done_pert:
                    break
            if not done_pert:
                res.append(tmp.cpu().detach().numpy())                
        res = np.array(res)
        res = torch.from_numpy(res).to('cuda')
        return res
    
    
class NormalAttack:
    def __init__(self, model, eps, method, stepsize, iters):
        self.model = model
        self.eps = eps
        self.method = method
        self.stepsize = stepsize
        self.iters = iters
        
    def normal_attack(self, images, labels):
        if self.method == 1:
            return self._method1_attack(images, labels)
        else:
            return self._method2_attack(images, labels)
        
    def _method1_attack(self, images, labels):
        # adding random noise to the data alone the normal direction
        D = images.shape[1]
        N = images.shape[0]
        if D == 2:
           res = []
            for d in images:
                rand_num = (np.random.random() * 2 * self.eps + (1 - self.eps)) ** 0.5
                res.append(d * rand_num)
            res = torch.stack(res)
        else:
            rand_pert = torch.randn(N, 1) * self.eps
            zero_array = torch.zeros(N, 2)
            rand_pert_final = torch.cat((zero_array, rand_pert), 1)
            rand_pert_final = rand_pert_final.to('cuda')
            res = images + rand_pert_final         
        return res
            
       
    def _method2_attack(self, images, labels):
        # grid search methods
        if images.shape[1] == 2:
            return self._method22d_attack(images, labels)
        else:
            return self._method23d_attack(images, labels)
        
    def _method22d_attack(self, images, labels):
        res = []
        search_range = [-1 * self.eps + i * 2 * self.eps for i in range(self.stepsize)]
        for data, target in zip(images, labels):
            max_diff = 0
            done_pert = False
            tmp = data
            for pert in search_range:
                adv = data * (1 - pert)
                adv = adv.to('cuda')
                loss = self.model(adv)
                label = (torch.sign(loss - 0.5) + 1) / 2 
                if label != target:
                    res.append(adv.cpu().detach().numpy())
                    done_pert = True
                    break
                else:
                    if torch.abs(loss - target) > max_diff:
                        tmp = adv
                        max_diff = torch.abs(loss - target)
            if not done_pert:
                res.append(tmp.cpu().detach().numpy())                
        res = np.array(res)
        res = torch.from_numpy(res).to('cuda')
        return res
 
    def _method23d_attack(self, images, labels):
        res = []
        search_range = [-1 * self.eps + i * 2 * self.eps for i in range(self.stepsize)]
        for data, target in zip(images, labels):
            max_diff = 0
            done_pert = False
            tmp = data
            for pert in search_range:
                adv = torch.Tensor(3)
                adv[0] = data[0]
                adv[1] = data[1]
                adv[2] = pert
                adv = adv.to('cuda')
                loss = self.model(adv)
                label = (torch.sign(loss - 0.5) + 1) / 2 
                if label != target:
                    res.append(adv.cpu().detach().numpy())
                    done_pert = True
                    break
                else:
                    if torch.abs(loss - target) > max_diff:
                        tmp = adv
                        max_diff = torch.abs(loss - target)
            if not done_pert:
                res.append(tmp.cpu().detach().numpy())                
        res = np.array(res)
        res = torch.from_numpy(res).to('cuda')
        return res
