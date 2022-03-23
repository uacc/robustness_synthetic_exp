from data_gen import *
from attack_method import *
from utils import *
from model import *
from mu_measure import *
import json
import argparse
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time
from matplotlib import pyplot as plt
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
from collections import defaultdict
from train_classifier import *

def train_normal_new(data, targets, hidden_layer, hidden_list, D, lr, wd, epochs, epsilon, normal_method, stepsize, iters):
    t0 = time()
    #hidden_layer = 2
    #hidden_list = [512, 256]
    org_net = Feedforward(D,hidden_layer, hidden_list).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(org_net.parameters(), lr=lr, weight_decay = wd)
    y_pred = org_net(data)
    org_data = data.clone()
    before_train = criterion(y_pred.squeeze(), targets)
    print('Test loss before training' , before_train.item())
    org_net.train()
    for epoch in range(epochs):
        normal = NormalAttack(org_net, epsilon, normal_method, stepsize, iters)
        data = normal.normal_attack(org_data, targets)
        optimizer.zero_grad()
        # Forward pass
        y_pred = org_net(data)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), targets).to(device)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Extra Training for original data
        optimizer.zero_grad()
        pgd = PGDAttack(org_net, epsilon, 0.01, iters)
        pgd_data = pgd.pgd_attack(org_data, targets)
        y_pred = org_net(pgd_data)
        loss = loss = criterion(y_pred.squeeze(), targets).to(device)
        loss.backward(retain_graph=True)
        optimizer.step()

    print("Training acc now " + str(acc(org_net, org_data, targets)))
    total_loss = loss.item()
    print('Training Done:{}s eclipsed.'.format(time()-t0))
    return org_net, total_loss

def main():  
    # add parser part later
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-layer', default = 2, type = int, help = 'number of hidden layer')
    parser.add_argument('--hidden-list', nargs='+', default = [512, 256], type = int, help = 'hidden layer width 512 256')
    parser.add_argument('--N', default =1000 , type = int, help = 'number of training sample')
    parser.add_argument('--D', default = 2, type = int, help = 'data dimension')
    parser.add_argument('--lr', default = 0.1, type = float, help = 'learning rate')
    parser.add_argument('--wd', default = 1e-3, type = float, help = 'weight decay')
    parser.add_argument('--epochs', default = 50, type = int, help = 'training epoch')
    parser.add_argument('--alpha', default = 0.01, type = float, help = 'perturbation step size')
    parser.add_argument('--epsilon-list', nargs = '+', default = [0.01, 0.03], type = float, help = 'epsilon list')
    parser.add_argument('--iters', default = 50, type = int, help = 'pgd perturbation iterations')
    parser.add_argument('--inmani-method', default = 1, type = int, help = 'inmanifold attack method')
    parser.add_argument('--normal-method', default = 1, type = int, help = 'normal attack method')
    parser.add_argument('--data-method', default = 1, type = int, help = 'data label methods')
    parser.add_argument('--eval-iters', default = 1, type = int, help = 'evaluation iterations')
    parser.add_argument('--perturb', default = 0., type = float, help = 'data label perturbation parameter')
    parser.add_argument('--stepsize', default = 10, type = int, help = 'number of step size for grid search method')
    parser.add_argument('--bd-data', default = 3000, type = int, help = 'number of data to draw boundary')
    parser.add_argument('--meas-step', default = 30, type = int, help = 'grid search in measure.')
    # para part
    args = parser.parse_args()
    hidden_layer = args.hidden_layer
    hidden_list = args.hidden_list
    N = args.N
    D = args.D
    meas_step = args.meas_step
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    alpha = args.alpha
    epsilon_list = args.epsilon_list
    iters = k = args.iters
    inmani_method = args.inmani_method
    normal_method = args.normal_method
    data_method = args.data_method
    eval_iters = args.eval_iters
    perturb = args.perturb
    stepsize = args.stepsize
    
#     gen_risk =defaultdict(list)
#     normal_risk = defaultdict(list)
#     in_adv_risk = defaultdict(list)
#     gen_adv_risk = defaultdict(list)
#     gen_meas =defaultdict(list)
#     normal_meas = defaultdict(list)
#     gen_adv_meas = defaultdict(list)
    bayes_risk = defaultdict(list)
    adv_risk = defaultdict(list)
    normal_risk = defaultdict(list)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    data_gen = toyData(N, D, perturb, data_method)
    
    
    for iterations in range(eval_iters):
        print("*****************************************************************")
        print("*************Starting " + str(iterations) + "th iteration.*************")
        print("*****************************************************************")
        print("\n")
        for epsilon in epsilon_list:
            print("################Current epsilon " + str(epsilon) + ".################")
            # generate original data and load to cuda
            print("============Training Orginal classifier.============")
            x_data, y_data = data_gen.generate()
            data = torch.FloatTensor(x_data).to(device)
            targets = torch.FloatTensor(y_data).to(device)
            
            # training original classifier
            B1, B1_loss = train_net(data, targets, hidden_layer, hidden_list, D, lr, wd, epochs)
            print(acc(B1, data, targets))
            
            # setup perturbation based on B1 model
            print("============Training PGD classifier. ============")
            # pgd = PGDAttack(B1, epsilon, alpha, iters)
            # inmani = InManiAttack(B1, epsilon, alpha, inmani_method, stepsize, iters)
            # double_inmani = InManiAttack(B1, epsilon * 2.0, alpha, inmani_method, stepsize, iters)
            # normal = NormalAttack(B1, epsilon, normal_method, stepsize, iters)
            
            
            # pgd attack data and training adv_train_pgd classifier as C3
            # attack_data = pgd_attack_images(B1, data, targets, epsilon, alpha, k)
            #attack_data = pgd.pgd_attack(data, targets)
            #merge_pgd, merge_targets = adv_train_set(data, attack_data, targets, targets)
            
            #C3, C3_loss = train_net(merge_pgd, merge_targets, hidden_layer, hidden_list, D, lr, wd, epochs)
            C3, C3_loss = train_pgd(data, targets, hidden_layer, hidden_list, D, lr, wd, epochs, epsilon, alpha, iters)

            
#             print("============Training inmanifold classifier. ============")
            # inmanifold attack data (2*epsilon) using project 
            # train in manifold classifier as C1
#             two_epsilon_attack_data = pgd_attack_images(B1, data, targets, 2 * epsilon, alpha, k)
#             project_adv_data = unit_proj(two_epsilon_attack_data, N, D)
#             project_adv_data = project_adv_data.to(device)
#             project_adv_data = double_inmani.inmani_attack(data, targets)
#             merge_in_manifold, merge_in_manifold_targets = adv_train_set(data, project_adv_data, targets, targets)
#             C1, C1_loss = train_net(merge_in_manifold, merge_in_manifold_targets, hidden_layer, hidden_list, D, lr, wd, epochs)

            # normal data and normal trained
#             x_data_normal = normal_data_1(0.1)
#             y_data_normal = gen_y(x_data_normal, 0)
#             data_normal = torch.FloatTensor(x_data_normal)
#             targets_normal = torch.FloatTensor(y_data_normal)
#             data_normal = data_normal.to(device)
#             targets_normal = targets_normal.to(device)
            # add gen_y_tensor
            print("============Training normal classifier. ============")
#             data_normal = normal.normal_attack(data, targets)
# #             if D == 2:
# #                 targets_normal = data_gen._gen_2D_y(data_normal)
# #                 targets_normal = torch.FloatTensor(targets_normal).to('cuda')
#             merge_normal_data, merge_normal_targets = adv_train_set(data,data_normal, targets, targets)
#             B2, B2_loss = train_net(merge_normal_data, merge_normal_targets, hidden_layer, hidden_list, D, lr, wd, epochs)
            #B2, B2_loss = train_normal(data, targets, hidden_layer, hidden_list, D, lr, wd, epochs, epsilon, 1, stepsize, iters)
            B2, B2_loss = train_normal_new(data, targets, hidden_layer, hidden_list, D, lr, wd, epochs, epsilon, 1, stepsize, iters)
    
            # test data on B1 attack also on B1
            # test data 
            print("============Evaluate B2 on test data set. ============")
            test_data, test_targets = data_gen.generate()
            test_data = torch.FloatTensor(test_data).to(device)
            test_targets = torch.FloatTensor(test_targets).to(device)
            pgd2 = PGDAttack(B2, epsilon, alpha, iters)
            #inmani = InManiAttack(B1, epsilon, alpha, inmani_method, stepsize, iters)
            double_inmani2 = InManiAttack(B2, epsilon * 2.0, alpha, inmani_method, stepsize, iters)
            normal2 = NormalAttack(B2, epsilon, normal_method, stepsize, iters)
            
            # test adv data
            # generate normal data
            
            test_norm_data = normal2.normal_attack(test_data, test_targets)
            #test_norm_targets = torch.FloatTensor(data_gen._gen_2D_y(test_norm_data)).to(device)
            test_norm_targets = test_targets
            print('generate inmanifold test attack')
            test_in_data = double_inmani2.inmani_attack(test_data, test_targets)
            test_in_targets = test_targets
            # generate pgd data
            print('generate pgd test attack')
            test_pgd_data = pgd2.pgd_attack(test_data, test_targets)
            test_pgd_targets = test_targets

            bayes_risk[str(epsilon)].append((1 - acc(B1, test_data, test_targets), 1 - acc(B1, test_pgd_data, test_pgd_targets), 1 - acc(B1, test_norm_data, test_norm_targets)))
            adv_risk[str(epsilon)].append((1 - acc(C3, test_data, test_targets), 1 - acc(C3, test_pgd_data, test_pgd_targets), 1 - acc(C3, test_norm_data, test_norm_targets)))
            normal_risk[str(epsilon)].append((1 - acc(B2, test_data, test_targets), 1 - acc(B2, test_pgd_data, test_pgd_targets), 1 - acc(B2, test_norm_data, test_norm_targets)))
            
            
    file_fold = './' + str(D) + 'd/'
    file_name = 'lable_' + str(data_method) + '_perturb_' + str(perturb)
    file_name += '_model_layer_' + str(hidden_layer) + '_max_hidden_' + str(max(hidden_list))
    file_name += '_in_attack_' + str(inmani_method) + '_norm_attack_' + str(normal_method)
    file_name += '_epsilon_' + str(epsilon)
    file_name += '_epochs_' + str(epochs) + '_lr_' + str(lr)
    file_name = file_fold + file_name + '_intuition'
    
    res_dic = {}
    res_dic['bayes_risk'] = bayes_risk
    res_dic['adv_risk'] = adv_risk
    res_dic['normal_risk'] = normal_risk
    
    if not os.path.isdir(file_fold):
        os.mkdir(file_fold)
    outfile = open(file_name + '_results.json', 'w')
    json.dump(res_dic, outfile, indent = 2)
    outfile.close()
    
    
if __name__ == "__main__":
    main()
          
