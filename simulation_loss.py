import fa_two_layer
import data_gen
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import fa_autograd
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 28})
# plt.rcParams["figure.figsize"] = (9, 9)
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)

seed0 = np.random.randint(100000)
seed1 = np.random.randint(100000)
seed2 = np.random.randint(100000)
n, d, p = (50, 10, 1000)
# X, y = data_gen.lr_data(n, d)
X, y = data_gen.rand_nn_data(n, d, p)
step = 10e-6
n_step = 100000
activation = 'relu'

net = fa_two_layer.TwoLayerNetwork(activation, d, p, n, seed1)
loss_bp = net.back_propagation(X, y, step, n_step)
loss_fa1, _, _ = net.feedback_alignment(
    X, y, step, regular='non', n_steps=n_step)
loss_fa10, _, _ = net.feedback_alignment(
    X, y, step, regular=10, n_steps=n_step)
plt.plot(np.arange(n_step)[:10000], loss_bp[:10000], np.arange(n_step)[
         :10000], loss_fa1[:10000], np.arange(n_step)[:10000], loss_fa10[:10000])


class TwoLayerFeedbackAlignmentNetworkReLU(nn.Module):
    def __init__(self, input_features, hidden_features):
        super(TwoLayerFeedbackAlignmentNetworkReLU, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentReLU(
            self.input_features, self.hidden_features)
        self.second_layer = fa_autograd.RegLinear(self.hidden_features, 1)

    def forward(self, X):
        X = torch.FloatTensor(X)
        hidden = self.first_layer(X)
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerBackPropNetworkReLU(nn.Module):
    def __init__(self, input_features, hidden_features):
        super(TwoLayerBackPropNetworkReLU, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = nn.Linear(input_features, hidden_features, bias=False)
        self.second_layer = nn.Linear(hidden_features, 1, bias=False)
        nn.init.normal_(self.first_layer.weight)
        nn.init.normal_(self.second_layer.weight)

    def forward(self, X):
        X = torch.FloatTensor(X)
        hidden = F.relu(self.first_layer(X))
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction

learning_rate = 10e-6
torch_net_fa = TwoLayerFeedbackAlignmentNetworkReLU(d, p)
optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=learning_rate)
loss_fn_fa = nn.MSELoss()
y_torch = torch.FloatTensor(y).unsqueeze(1)
loss_fa_autograd = []
for t in range(n_step):
    pred = torch_net_fa.forward(X)
    loss = loss_fn_fa(pred, y_torch)
    loss_fa_autograd.append(loss.item())
    if t % 100 == 99:
        print(t, loss.item())
    optimizer_fa.zero_grad()
    loss.backward()
    optimizer_fa.step()

plt.plot(np.arange(n_step)[:10000], loss_fa_autograd[:10000])


torch_net_bp = TwoLayerBackPropNetworkReLU(d, p)
optimizer_bp = torch.optim.SGD(torch_net_bp.parameters(), lr=learning_rate)
loss_fn_bp = nn.MSELoss()
y_torch = torch.FloatTensor(y).unsqueeze(1)
y_torch.size()
loss_bp_autograd = []
for t in range(n_step):
    pred = torch_net_bp.forward(X)
    loss = loss_fn_bp(pred, y_torch)
    loss_bp_autograd.append(loss.item())
    if t % 100 == 99:
        print(t, loss.item())
    optimizer_bp.zero_grad()
    loss.backward()
    optimizer_bp.step()

plt.plot(np.arange(n_step)[:1000], loss_bp_autograd[:1000], np.arange(
    n_step)[:1000], loss_fa_autograd[:1000])
