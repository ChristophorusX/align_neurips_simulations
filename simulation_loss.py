import fa_two_layer
import data_gen
import numpy as np
import torch
from torch import nn
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
n_step = 10000
activation = 'relu'

net = fa_two_layer.TwoLayerNetwork(activation, d, p, n, seed1)
loss_bp = net.back_propagation(X, y, step, n_step)
loss_fa1, _, _ = net.feedback_alignment(
    X, y, step, regular='non', n_steps=n_step)
loss_fa10, _, _ = net.feedback_alignment(
    X, y, step, regular=10, n_steps=n_step)
plt.plot(np.arange(n_step)[:1000], loss_bp[:1000], np.arange(n_step)[
         :1000], loss_fa1[:1000], np.arange(n_step)[:1000], loss_fa10[:1000])


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
        prediction = self.second_layer(hidden)
        return prediction


torch_net = TwoLayerFeedbackAlignmentNetworkReLU(d, p)
optimizer = torch.optim.SGD(torch_net.parameters(), lr=10e-6)
loss_fn = nn.MSELoss()
y = torch.FloatTensor(y).unsqueeze(1)

for t in range(n_step):
    pred = torch_net.forward(X)
    loss = loss_fn(pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
