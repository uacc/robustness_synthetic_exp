# robust_toy
toy data experiments for manifold robust learning

file name system

data dimension -> folder

data label methods -> 1: label flip 2: seperation

model info: number of layers and max hidden layers

attack parameters: epsilon, in-manifold methods, normal methods

training: epochs, learning rate

save decision boundary as db

save json file with name result


## Theorem 1 part 1
To compute measure, we sort the point with non-zero normal risk based on the classifier and compute the 2epsilon length (2d case) or 2epsilon area (3d case) as the measure.

How many classifier do we need? Bayesian optimal, normal classifier and adversarial classifier???

## Theorem 1 part 2
2d case almost done.

boundary
![alt text](https://github.com/uacc/robust_toy/blob/master/2d/lable_2_perturb_0.01_model_layer_2_max_hidden_512_in_attack_1_norm_attack_1_epsilon_0.1_epochs_5000_lr_0.1_bd.png)
risk
![alt_text](https://github.com/uacc/robust_toy/blob/master/2d/lable_2_perturb_0.01_model_layer_2_max_hidden_512_in_attack_1_norm_attack_1_epsilon_start_0.01_end_0.3_epochs_5000_lr_0.1_risk.png)

3d case, need larger epsilon to see the problem

boundary (original label)
![alt_text](https://github.com/uacc/robust_toy/blob/master/3d/lable_1_perturb_0.0_model_layer_2_max_hidden_512_in_attack_1_norm_attack_1_epsilon_1.5_epochs_3000_lr_0.1_bd.png)

risk
![alt_text](https://github.com/uacc/robust_toy/blob/master/3d/lable_1_perturb_0.0_model_layer_2_max_hidden_512_in_attack_1_norm_attack_1_epsilon_start_0.01_end_0.3_epochs_3000_lr_0.1_risk.png)

3d with flipped label
boundary
![alt_text](https://github.com/uacc/robust_toy/blob/master/3d/lable_2_perturb_0.01_model_layer_2_max_hidden_512_in_attack_1_norm_attack_1_epsilon_0.3_epochs_3000_lr_0.1_bd.png)
risk
![alt_text](https://github.com/uacc/robust_toy/blob/master/3d/lable_2_perturb_0.01_model_layer_2_max_hidden_512_in_attack_1_norm_attack_1_epsilon_start_0.01_end_1.5_epochs_3000_lr_0.1_risk.png)
Need to fine tune the network, right now the normal accuracy is not good enough.
