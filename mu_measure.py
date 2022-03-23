import numpy as np
import math
import torch

# Segment Tree for 3d case
class Node:
    def __init__(self, start, end, X):
        self.start, self.end = start, end
        self.total = self.count = 0
        self._left = self._right = None
        self.X = X

    @property
    def mid(self):
        return (self.start + self.end) // 2

    @property
    def left(self):
        self._left = self._left or Node(self.start, self.mid, self.X)
        return self._left

    @property
    def right(self):
        self._right = self._right or Node(self.mid, self.end, self.X)
        return self._right

    def update(self, i: int, j: int, val: int) -> int:
        if i >= j:
            return 0
        if self.start == i and self.end == j:
            self.count += val
        else:
            self.left.update(i, min(self.mid, j), val)
            self.right.update(max(self.mid, i), j, val)

        if self.count > 0:
            self.total = self.X[self.end] - self.X[self.start]
        else:
            self.total = self.left.total + self.right.total

        return self.total

def rectangleArea(rectangles):
    OPEN, CLOSE = 1, -1
    events = []

    X = set()
    for x1, y1, x2, y2 in rectangles:
        if (x1 < x2) and (y1 < y2):
            events.append((y1, OPEN, x1, x2))
            events.append((y2, CLOSE, x1, x2))
            X.add(x1)
            X.add(x2)
    events.sort()

    X = sorted(X)
    x_index = {x: i for i, x in enumerate(X)}
    active = Node(0, len(X) - 1, X)
    ans = 0
    cur_x_sum = 0
    cur_y = events[0][0]

    for y, typ, x1, x2 in events:
        ans += cur_x_sum * (y - cur_y)
        cur_x_sum = active.update(x_index[x1], x_index[x2], typ)
        cur_y = y

    return ans

def get_meas(net, data, targets, meas_step, epsilon):
    if data.shape[1] == 2:
        return _get_2d_meas(net, data, targets, meas_step, epsilon)
    else:
        return _get_3d_meas(net, data, targets, meas_step, epsilon)
    

def _get_2d_meas(net, data, targets, meas_step, epsilon):
    meas = 0
    # list element in Z
    norm_data_ori = _grid_2d_norm(net, data, targets, meas_step, epsilon)
    if norm_data_ori.shape[0] == 0:
        return 0
    # sort index
    norm_data = norm_data_ori[norm_data_ori[:, 0].argsort()]
    
    # compute segment
    i, j = 0, 0
    sec = {}
    k = 0
    while i < norm_data.shape[0]:
        sec[k] = []
        while j < norm_data.shape[0] and np.abs(norm_data[i][1] - norm_data[j][1]) < epsilon and np.abs(norm_data[i][2] - norm_data[j][2]) < epsilon:
            sec[k].append(norm_data[j])
            i = j
            j += 1
        k += 1
        i = j

    circular = False
    if np.abs(norm_data[0][1] - norm_data[-1][1]) < epsilon and np.abs(norm_data[0][2] - norm_data[-1][2]) < epsilon:
        meas += length(sec[0][-1], sec[k-1][0], epsilon)
        circular = True
    for i in range(k):
        if (i == 0 or i == k - 1) and circular:
            continue
        else:
            meas += length(sec[i][-1], sec[i][0], epsilon)
    
    return meas / (2 * math.pi)


def length(head, tail, epsilon):
    # input of head and tail to be [angle, x, y] form numpy array
    if head[1] > 0 and head[2] > 0:
        angle_head = theta_fun(head[1] - epsilon, head[2] + epsilon)
    elif head[1] < 0 and head[2] > 0:
        angle_head = theta_fun(head[1] - epsilon, head[2] - epsilon)
    elif head[1] < 0 and head[2] < 0:
        angle_head = theta_fun(head[1] + epsilon, head[2] - epsilon)
    else:
        angle_head = theta_fun(head[1] + epsilon, head[2] + epsilon)
        
    if tail[1] > 0 and tail[2] > 0:
        angle_tail = theta_fun(tail[1] + epsilon, tail[2] - epsilon)
    elif tail[1] < 0 and tail[2] > 0:
        angle_tail = theta_fun(tail[1] + epsilon, tail[2] + epsilon)
    elif tail[1] < 0 and tail[2] < 0:
        angle_tail = theta_fun(tail[1] - epsilon, tail[2] + epsilon)
    else:
        angle_tail = theta_fun(tail[1] - epsilon, tail[2] - epsilon)
    angle_diff = angle_head - angle_tail
    if angle_diff < 0:
        angle_diff += 2 * math.pi
    return angle_diff


def _grid_2d_norm(net, data, targets, meas_step, epsilon):
    # input ori data and targets on cude
    # output data with norm adv in numpy 
    # with format (theta, x, y)
    res = []
    search_range = [(1 - epsilon) + i * 2 * epsilon / meas_step for i in range(meas_step)]
    for d, t in zip(data, targets):
        for i in search_range:
            adv_data = d * i
            label = (torch.sign(net(adv_data)- 0.5) + 1) / 2 
            if label != t:
                x = d[0].cpu().detach().numpy()
                y = d[1].cpu().detach().numpy()
                res.append([theta_fun(x, y), x, y])
                break
    res = np.array(res)
    return res


def theta_fun(x, y):
    # output angle from 0 to 2pi
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += math.pi * 2
    return angle


def _get_3d_meas(net, data, targets, meas_step, epsilon):
    meas = 0
    
    # list element in Z
    norm_data = _grid_3d_norm(net, data, targets, meas_step, epsilon)
    if norm_data.shape[0] == 0:
        return 0
    
    # sort index
    rectangle = [[d[0]-epsilon, d[1] - epsilon, d[0] + epsilon, d[1] + epsilon] for d in norm_data]
    rectangle = np.array(rectangle)
    rectangle_sorted = rectangle[rectangle[:,0].argsort(axis = 0)]

    meas = rectangleArea(rectangle_sorted)
    
    return meas / (4 * math.pi * math.pi)


def  _grid_3d_norm(net, data, targets, meas_step, epsilon):
    res = []
    search_range = [-1 * epsilon + 2 * epsilon * i / meas_step for i in range(meas_step)]
    for d, t in zip(data, targets):
        for i in search_range:
            adv_data = d.clone()
            adv_data[2] = i
            label = (torch.sign(net(adv_data)- 0.5) + 1) / 2 
            if label != t:
                res.append(adv_data.cpu().detach().numpy())
                break
    res = np.array(res)
    return res

