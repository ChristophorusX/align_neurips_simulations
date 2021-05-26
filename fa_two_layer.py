import torch
import torch.nn as nn
import numpy as np


# Construct a neural network with manual parameter update
class TwoLayerNetwork(object):
    def __init__(self, activation, dim_input, dim_hidden, dim_output, seed):
        # Initialize dimensions
        self.d = dim_input
        self.n = dim_output
        self.p = dim_hidden
        self.seed = seed
        # Initialize parameters
        np.random.seed(self.seed)
        self.W0 = torch.FloatTensor(np.random.randn(self.d, self.p))
        self.beta0 = torch.FloatTensor(np.random.randn(self.p, 1))
        # Initialize activation function
        if activation == 'non':
            self.activation = lambda x: x
            self.act_derivative = lambda x: torch.ones_like(x)
        elif activation == 'relu':
            self.activation = torch.relu
            self.act_derivative = lambda x: torch.relu(torch.sign(x))
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
            self.act_derivative = lambda x: torch.sigmoid(
                x)(1 - torch.sigmoid(x))
        elif activation == 'tanh':
            self.activation = torch.tanh
            self.act_derivative = lambda x: torch.ones_like(x) - torch.tanh(x) * torch.tanh(x)

    def forward(self, X):
        if hasattr(self, 'W') and hasattr(self, 'beta'):
            with torch.no_grad():
                X = torch.FloatTensor(X)
                self.H = torch.matmul(X, self.W)
                self.H_activated = self.activation(self.H)
                self.f = torch.matmul(self.H_activated, self.beta) / np.sqrt(self.p)
        else:
            with torch.no_grad():
                X = torch.FloatTensor(X)
                self.H = torch.matmul(X, self.W0)
                self.H_activated = self.activation(self.H)
                self.f = torch.matmul(self.H_activated, self.beta0) / np.sqrt(self.p)
        return self.f.data.numpy().flatten()

    def back_propagation(self, X, y, step, n_steps=10000):
        # Load training data
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)
        # Load init parameters
        self.W = self.W0.detach().clone()
        self.beta = self.beta0.detach().clone()
        # Backpropagate
        loss = []
        for t in np.arange(n_steps):
            with torch.no_grad():
                self.H = torch.matmul(X, self.W)
                self.H_activated = self.activation(self.H)
                self.f = torch.matmul(
                    self.H_activated, self.beta) / np.sqrt(self.p)
                current_loss = 0.5 * torch.sum((y - self.f)**2) / np.sqrt(self.n)
                loss.append(current_loss)
                if t % (n_steps / 5) == 0:
                    print("iteration %d: TRAINING loss %f" % (t, current_loss))
                e = self.f - y
                grad_beta = torch.matmul(
                    torch.transpose(self.H_activated, 0, 1), e) / np.sqrt(self.p)
                mask = self.act_derivative(self.H)
                V = torch.matmul(e, self.beta.view(1, -1))
                V_tilde = mask * V
                grad_W = torch.matmul(torch.transpose(
                    X, 0, 1), V_tilde) / np.sqrt(self.p)
                self.W += -step * grad_W
                self.beta += -step * grad_beta
        return loss

    def feedback_alignment(self, X, y, step, regular='non', n_steps=10000):
        # Initialize random backpropagation weights
        self.b = torch.FloatTensor(np.random.randn(self.p, 1))
        # Load training data
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)
        # Load init parameters
        self.W = self.W0.detach().clone()
        self.beta = self.beta0.detach().clone()
        # Initialize regularization
        if regular == 'non':
            reg = 0
        else:
            reg = regular
        # Feedback alignment
        loss = []
        for t in np.arange(n_steps):
            with torch.no_grad():
                self.H = torch.matmul(X, self.W)
                self.H_activated = self.activation(self.H)
                self.f = torch.matmul(
                    self.H_activated, self.beta) / np.sqrt(self.p)
                current_loss = 0.5 * torch.sum((y - self.f)**2) / np.sqrt(self.n)
                loss.append(current_loss)
                if t % (n_steps // 5) == 0:
                    print("iteration %d: TRAINING loss %f" % (t, current_loss))
                e = self.f - y
                grad_beta = torch.matmul(
                    torch.transpose(self.H_activated, 0, 1), e) / np.sqrt(self.p)
                mask = self.act_derivative(self.H)
                V = torch.matmul(e, self.b.view(1, -1))
                V_tilde = mask * V
                grad_W = torch.matmul(torch.transpose(
                    X, 0, 1), V_tilde) / np.sqrt(self.p)
                self.W += -step * grad_W
                self.beta += -step * grad_beta - step * reg * self.beta
        return loss, self.beta.data.numpy().flatten(), self.b.data.numpy().flatten()


if __name__ == '__main__':
    seed = np.random.randint(100000)
    net = TwoLayerNetwork('relu', 20, 200, 100, seed)
    net.forward(np.random.rand(100, 20))
