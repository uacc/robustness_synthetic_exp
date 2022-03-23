import numpy as np
import math

class toyData:
    def __init__(self, N, D, perturb, method):
        self.N = N
        self.D = D
        self.perturb = perturb
        self.method = method
    
    def generate(self):
        # generate 2d and 3d synthetic dataset 
        if self.D == 2:
            x_data = self._gen_2D_x(self.N, self.D)
            if self.perturb != 0:
                if self.method == 1:
                    y_data = self._gen_2D_y_method1(x_data)
                else:
                    y_data = self._gen_2D_y_method2(x_data)
            else:
                y_data = self._gen_2D_y(x_data)
        else:
            x_data = self._gen_3D_x(self.N, self.D)
            if self.method != 0:
                if self.method == 1:
                    y_data = self._gen_3D_y_method1(x_data)
                else:
                    y_data = self._gen_3D_y_method2(x_data)
            else:
                y_data = self._gen_3D_y(x_data)
        return x_data, y_data
    
    def _gen_2D_x(self, N, D):
        # generate 2d data on unit circle
        res = []
        for i in range(N):
            theta = 2 * np.pi * np.random.random()
            u = np.array([np.sin(theta), np.cos(theta)])
            denom = (np.sum(u**2))**0.5
            res.append(u/denom)
        res = np.array(res)
        return res
    
    def _gen_3D_x(self, N, D):
        # generate 3d data on xy-plane separated by sin(x) 
        res = []
        for i in range(N):
            x1 = np.random.random() * 2 * np.pi - np.pi
            x2 = np.random.random() * 2 * np.pi - np.pi
            u = np.array([x1, x2, 0])
            res.append(u)
        res = np.array(res)
        return res
        

    def _gen_2D_y(self, x_data):
        # assign lable to 2d data
        res = [1. if x[0] >= 0 else 0. for x in x_data]
        res = np.array(res)
        return res
    
    def _gen_3D_y(self, x_data):
        # assign label to 3d data
        res = [1. if x[0] >= math.sin(x[1]) else 0. for x in x_data]
        res = np.array(res)
        return res
    
    def _gen_2D_y_method1(self, x_data):
        # generate 2d  data with label noise
        res = []
        for x in x_data:
            rand_sign = np.sign(np.random.random_sample() - self.perturb)
            if x[0]>= 0:
                res.append(1. * rand_sign)
            else:
                res.append(-1. * rand_sign)
        res = [x if x == 1 else 0 for x in res]
        res = np.array(res)
        return res
    
    def _gen_3D_y_method1(self, x_data):
        # generate 3d data with label noise
        res = []
        for x in x_data:
            rand_sign = np.sign(np.random.random_sample() - self.perturb)
            if x[0] >= math.sin(x[1]):
                res.append(1 * rand_sign)
            else:
                res.append(-1. * rand_sign)
        res = [x if x == 1 else 0 for x in res]
        res = np.array(res)
        return res
    
    def _gen_2D_y_method2(self, x_data):
        # generate 2d data with double decision boundary
        res = []
        for x in x_data:
            if x[0] >= 0 and x[1] >= 0:
                res.append(1.)
            elif x[0] >=0 and x[1] < 0:
                res.append(0.)
            elif x[0] < 0 and x[1] < 0:
                res.append(1.)
            else:
                res.append(0.)
        res = np.array(res)
        return res
    
    def _gen_3D_y_method2(self, x_data):
        # generate 3d data with double decision boundary
        res = []
        for x in x_data:
            if x[0] >= math.sin(x[1]) and x[1] >= 0:
                res.append(1.)
            elif x[0] >= math.sin(x[1]) and x[1] < 0:
                res.append(0.)
            elif x[0] < math.sin(x[1]) and x[1] < 0:
                res.append(1.)
            else:
                res.append(0.)
        res = np.array(res)
        return res
