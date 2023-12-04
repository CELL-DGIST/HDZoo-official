"""
TrainableHD - Yeseong Kim & Jiseung Kim (CELL) @ DGIST, 2023
"""
import torch


class Adam:
    def __init__(self):
        self.m = None
        self.v = None
        self.t = 0

    def optimize(self, grad, lr):
        self.t += 1
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * torch.pow(grad, 2)
        return lr * self.m / (1 - 0.9 ** self.t)/ (1e-8 + self.v / (1 - 0.999 ** self.t)) ** 0.5


class SGD:
    def optimize(self, grad, lr):
        return lr * grad


class Momentum:
    def __init__(self):
        self.m = None
        self.v = None
        self.t = 0

    def optimize(self, grad, lr, momemtum = 0.9):
        self.t += 1
        if self.m is None:
            self.m = torch.ones(grad.shape) * momemtum
            self.v = torch.zeros_like(grad)

            if torch.cuda.is_available():
                self.m = self.m.cuda()

        self.v = self.m * self.v - lr*grad
        return -1 * self.v


class NAG:
    def __init__(self):
        self.m = None
        self.v = None
        self.t = 0

    def optimize(self, grad, lr, momemtum = 0.9):
        self.t += 1
        if self.m is None:
            self.m = torch.ones(grad.shape) * momemtum
            self.v = torch.zeros_like(grad)

            if torch.cuda.is_available():
                self.m = self.m.cuda()

        self.v = self.m * self.v - lr*grad
        return -1 * (self.m * self.v - lr*grad)


class Adagrad:
    def __init__(self):
        self.g = None
        self.t = 0

    def optimize(self, grad, lr):
        self.t += 1
        if self.g is None:
            self.g = torch.zeros_like(grad)

        self.g = self.g + torch.pow(grad, 2)
        return lr * (torch.pow(self.g + 0.0001, -1/2))*grad #0.0001 is epsilon, to avoid deviding by zero


class RMS:
    def __init__(self):
        self.g = None
        self.t = 0

    def optimize(self, grad, lr):
        self.t += 1
        if self.g is None:
            self.g = torch.zeros_like(grad)

        self.g = 0.9*self.g + 0.1*torch.pow(grad, 2)
        return lr * (torch.pow(self.g + 0.0001, -1/2))*grad #0.0001 is epsilon, to avoid deviding by zero


class AdaDelta:
    def __init__(self):
        self.g = None
        self.w = None
        self.s = None
        self.t = 0

    def optimize(self, grad, lr):
        self.t += 1
        if self.g is None:
            self.g = torch.zeros_like(grad)
            self.w = torch.zeros_like(grad)
            self.s = torch.zeros_like(grad)

        self.g = 0.9*self.g + 0.1*torch.pow(grad, 2)
        self.w = torch.pow(self.s + 0.0001, 1/2)*torch.pow(self.g + 0.0001, -1/2)*grad
        self.s = 0.9*self.s + 0.1*torch.pow(self.w, 2)
        return lr * self.w 


""" Choose Optimizer """
def choose_optim(optimizer):
    if optimizer == 'AdaDelta':
        optim = AdaDelta()
    elif optimizer == 'RMS':
        optim = RMS()
    elif optimizer == 'Adagrad':
        optim = NAG()
    elif optimizer == 'Momentum':
        optim = Momentum()
    elif optimizer == 'SGD':
        optim = SGD()
    elif optimizer == 'Adam':
        optim = Adam()
    elif optimizer == 'NAG':
        optim = NAG()

    return optim 