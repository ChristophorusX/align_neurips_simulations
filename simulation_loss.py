import fa
import data_gen
import numpy as np
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

net = fa.TwoLayerNetwork(activation, d, p, n, seed1)
loss_bp = net.back_propagation(X, y, step, n_step)
loss_fa1 = net.feedback_alignment(X, y, step, regular='non', n_steps=n_step)
loss_fa10 = net.feedback_alignment(X, y, step, regular=10, n_steps=n_step)
plt.plot(np.arange(n_step)[:1000], loss_bp[:1000], np.arange(n_step)[
         :1000], loss_fa1[:1000], np.arange(n_step)[:1000], loss_fa10[:1000])
