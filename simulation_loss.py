import fa_two_layer
import data_gen
import net_autograd
import numpy as np
import torch
from torch import nn
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
step = 10e-5
n_step = 10000
activation = 'relu'

net = fa_two_layer.TwoLayerNetwork(activation, d, p, n, seed1)
loss_bp = net.back_propagation(X, y, step, n_step)
loss_fa1, _, _ = net.feedback_alignment(
    X, y, step, regular='non', n_steps=n_step)
loss_fa10, _, _ = net.feedback_alignment(
    X, y, step, regular=10, n_steps=n_step)
plt.plot(np.arange(n_step)[:10000], loss_bp[:10000], np.arange(n_step)[
         :10000], loss_fa1[:10000], np.arange(n_step)[:10000], loss_fa10[:10000])


learning_rate = 10e-5
reg = 0
torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkReLU(d, p, reg)
optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=learning_rate)
loss_fn_fa = nn.MSELoss()
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y).unsqueeze(1)
loss_fa_autograd = []
for t in range(n_step):
    pred = torch_net_fa.forward(X_torch)
    loss = loss_fn_fa(pred, y_torch)
    loss_fa_autograd.append(loss.item())
    if t % 100 == 99:
        print(t, loss.item())
    optimizer_fa.zero_grad()
    loss.backward()
    optimizer_fa.step()

plt.plot(np.arange(n_step)[:10000], loss_fa_autograd[:10000])


torch_net_bp = net_autograd.TwoLayerBackPropNetworkReLU(d, p)
optimizer_bp = torch.optim.SGD(torch_net_bp.parameters(), lr=learning_rate)
loss_fn_bp = nn.MSELoss()
y_torch = torch.FloatTensor(y).unsqueeze(1)
y_torch.size()
loss_bp_autograd = []
for t in range(n_step):
    X = torch.FloatTensor(X)
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


drop_rate = 0.1
torch_net_dropout = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkReLU(
    d, p, drop_rate)
optimizer_dropout = torch.optim.SGD(
    torch_net_dropout.parameters(), lr=learning_rate)
loss_fn_dropout = nn.MSELoss()
y_torch = torch.FloatTensor(y).unsqueeze(1)
y_torch.size()
loss_dropout_autograd = []
for t in range(n_step):
    X = torch.FloatTensor(X)
    pred = torch_net_dropout.forward(X)
    loss = loss_fn_dropout(pred, y_torch)
    loss_dropout_autograd.append(loss.item())
    if t % 100 == 99:
        print(t, loss.item())
    optimizer_dropout.zero_grad()
    loss.backward()
    optimizer_dropout.step()

plt.plot(np.arange(n_step)[:10000], loss_dropout_autograd[:10000], np.arange(n_step)[:10000], loss_fa_autograd[:10000])
