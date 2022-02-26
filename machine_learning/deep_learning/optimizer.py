from math import sqrt
import numpy as np
from abc import ABC, abstractmethod


class SearchMethod(ABC):
    epsilon = 1e-16

    @abstractmethod
    def update(self, x, y, x_slope, y_slope):
        pass


class GradientDescent(SearchMethod):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def update(self, x, y, x_slope, y_slope):
        x = x - x_slope * self.learning_rate
        y = y - y_slope * self.learning_rate
        return x, y


class Adam(SearchMethod):
    def __init__(self, beta1=0.9, beta2=0.99, eta=0.01):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.i = 1
        self.xm, self.ym, self.xv, self.yv = 0, 0, 0, 0

    def update(self, x, y, x_slope, y_slope):
        self.xm = self.beta1 * self.xm + (1 - self.beta1) * x_slope
        self.ym = self.beta1 * self.ym + (1 - self.beta1) * y_slope
        self.xv = self.beta2 * self.xv + (1 - self.beta2) * x_slope ** 2
        self.yv = self.beta2 * self.yv + (1 - self.beta2) * y_slope ** 2

        xmhat, ymhat = self.xm / (1 - self.beta1 ** self.i), self.ym / (1 - self.beta1 ** self.i)
        xvhat, yvhat = self.xv / (1 - self.beta2 ** self.i), self.yv / (1 - self.beta2 ** self.i)
        x = x - self.eta * (xmhat / sqrt(xvhat) + self.epsilon)
        y = y - self.eta * (ymhat / sqrt(yvhat) + self.epsilon)
        self.i += 1

        return x, y


class LowestDistance:
    epsilon = 1e-8

    def __init__(self, points):
        self.points = points

    def calculate_object(self, x, y):
        return sum([self.get_distance(x, y, xi, yi) for xi, yi in self.points])

    @classmethod
    def get_distance(cls, x, y, xi, yi):
        return sqrt((x - xi) ** 2 + (y - yi) ** 2) + cls.epsilon

    def optimizer(self, x, y, search_method: SearchMethod, threshold=0.00001):
        i = 0
        pre_object = self.calculate_object(x, y)
        while True:
            x_slope = sum([1 / self.get_distance(x, y, xi, yi) * (x - xi) for xi, yi in self.points])
            y_slope = sum([1 / self.get_distance(x, y, xi, yi) * (y - yi) for xi, yi in self.points])
            x, y = search_method.update(x, y, x_slope, y_slope)
            new_object = self.calculate_object(x, y)
            # print(x, y, x_slope, y_slope, new_object)
            if abs(pre_object - new_object) < threshold:
                return x, y, i
            else:
                pre_object = new_object
            i += 1


if __name__ == '__main__':
    points = [(0, 0), (1, 2), (2, 6), (4, 4), (3, 6)]
    ld = LowestDistance(points=points)
    learning_rates = np.arange(0.1, 0.5, 0.1)
    x0, y0 = 0, 0
    # As learning rate increase, the efficiency of gradient descend firstly increases and then decreases due to
    # overshot issue
    print('Change learning rate')
    for rl in learning_rates:
        search_method = GradientDescent(learning_rate=rl)
        print(ld.optimizer(x=x0, y=y0, search_method=search_method))

    thresholds = np.exp(np.arange(-7, -3))
    print('Change threshold')
    for ts in thresholds:
        search_method = GradientDescent(learning_rate=0.2)
        print(ld.optimizer(x=x0, y=y0, search_method=search_method, threshold=ts))

    print('Adam: change learning rate')
    etas = np.arange(0.05, 0.3, 0.05)
    for eta in etas:
        adam = Adam(eta=eta)
        print(ld.optimizer(x=x0, y=y0, search_method=adam))
