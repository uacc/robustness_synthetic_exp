# A Manifold View of Adversarial Risk

## Description
In this project, we verfies our theoretical results in our paper "A Manifold View of Adversarial Risk".
We testify our results on the following synthetic dataset.
![ALt text](./figs/synthetic_dataset.jpg?raw=ture "Datasets")
To verify our upper bound, we compare the adversarial risk and its decomposition on different classifiers with various epsilon value. Here is an example of comparing general adversarial risk and decomposition upper bound on standard classifier.

## How to run
To repeat our empirical results in the paper, run the following code
```python 
# test on single decision boundary 2d data
python measure_eval.py --D 2 --data-method 0 --eval-iters 10 --meas-step 50
# test on double decision boundary 2d data
python measure_eval.py --D 2 --data-method 2 --eval-iters 10 --meas-step 50
# 3d single boundary
python measure_eval.py --D 3 --data-method 0 --eval-iters 10 --meas-step 50
# 3d double boundary
python measure_eval.py --D 3 --data-method 2 --eval-iters 10 --meas-step 50
```
We also implemented the grid search methods to find in-manifold and normal perturbed data. Use the following code call grid search methods.
```python 
python measure_eval.py --D 2 --data-method 0 --inmani-method 2 --normal-method 2 --eval-iters 10 --meas-step 50
```
