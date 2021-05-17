import fa_two_layer
import net_autograd
import data_gen
import numpy as np
import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 28})
# plt.rcParams["figure.figsize"] = (9, 9)
# rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_align_df(n, d, p_list, reg_list, activation, synthetic_data, step, n_step, n_iter):
    reg_align_df = []
    for reg in reg_list:
        align_table = []
        for p in p_list:
            align_array = []
            for iter in range(n_iter):
                if synthetic_data == 'lr':
                    X, y = data_gen.lr_data(n, d)
                elif synthetic_data == 'nn':
                    X, y = data_gen.rand_nn_data(n, d, p)
                seed = np.random.randint(100000)
                net = fa_two_layer.TwoLayerNetwork(activation, d, p, n, seed)
                loss_fa, beta, b = net.feedback_alignment(
                    X, y, step, regular=reg, n_steps=n_step)
                align = np.inner(beta, b) / \
                    np.linalg.norm(beta) / np.linalg.norm(b)
                align_array.append(align)
            align_table.append(align_array)
        align_table = np.array(align_table)
        flattened_table = align_table.flatten()
        p_index = np.repeat(p_list, n_iter)
        reg_index = np.repeat(reg, len(p_list) * n_iter)
        activation_index = np.repeat(activation, len(p_list) * n_iter)
        combined_table = np.vstack(
            (p_index, flattened_table, reg_index, activation_index)).T
        align_df = pd.DataFrame(data=combined_table, columns=[
                                r"$p$ Hidden Layer Width", 'Alignment', r"Regularization $\lambda$", 'Activation'])
        align_df[r"$p$ Hidden Layer Width"] = align_df[r"$p$ Hidden Layer Width"].astype(
            int)
        align_df['Alignment'] = align_df['Alignment'].astype(np.double)
        align_df[r"Regularization $\lambda$"] = align_df[r"Regularization $\lambda$"].astype(
            int)
        reg_align_df.append(align_df)
    df = pd.concat(reg_align_df)
    return df


n, d, p = (50, 10, 1000)
# X, y = data_gen.lr_data(n, d)
X, y = data_gen.rand_nn_data(n, d, p)
step = 10e-6
n_step = 100000


def get_autograd_align_df(n, d, p_list, reg_list, activation, synthetic_data, step, n_step, n_iter):
    reg_align_df = []
    for reg in reg_list:
        align_table = []
        for p in p_list:
            align_array = []
            for iter in range(n_iter):
                if synthetic_data == 'lr':
                    X, y = data_gen.lr_data(n, d)
                elif synthetic_data == 'nn':
                    X, y = data_gen.rand_nn_data(n, d, p)
                if activation == 'relu':
                    torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkReLU(
                        d, p, reg)
                elif activation == 'non':
                    torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkLinear(d, p, reg)
                torch_net_fa.to(device)
                optimizer_fa = torch.optim.SGD(
                    torch_net_fa.parameters(), lr=step)
                loss_fn_fa = nn.MSELoss()
                y_torch = torch.FloatTensor(y).unsqueeze(1).to(device)
                for t in range(n_step):
                    X_torch = torch.FloatTensor(X).to(device)
                    pred = torch_net_fa.forward(X_torch)
                    loss = loss_fn_fa(pred, y_torch)
                    if t % (n_step / 5) == n_step / 5 - 1:
                        print(t, loss.item())
                    optimizer_fa.zero_grad()
                    loss.backward()
                    optimizer_fa.step()
                for name, param in torch_net_fa.named_parameters():
                    if name == 'second_layer.backprop_weight':
                        backprop_weight = param.data
                    if name == 'second_layer.weight':
                        second_layer_weight = param.data
                align = torch.tensordot(backprop_weight, second_layer_weight) / \
                    torch.norm(backprop_weight) / \
                    torch.norm(second_layer_weight)
                align = align.cpu().data.detach().numpy().flatten()
                print(align)
                align_array.append(align)
            align_table.append(align_array)
        align_table = np.array(align_table)
        flattened_table = align_table.flatten()
        p_index = np.repeat(p_list, n_iter)
        reg_index = np.repeat(reg, len(p_list) * n_iter)
        activation_index = np.repeat(activation, len(p_list) * n_iter)
        combined_table = np.vstack(
            (p_index, flattened_table, reg_index, activation_index)).T
        align_df = pd.DataFrame(data=combined_table, columns=[
                                r"$p$ Hidden Layer Width", 'Alignment', r"Regularization $\lambda$", 'Activation'])
        align_df[r"$p$ Hidden Layer Width"] = align_df[r"$p$ Hidden Layer Width"].astype(
            int)
        align_df['Alignment'] = align_df['Alignment'].astype(np.double)
        align_df[r"Regularization $\lambda$"] = align_df[r"Regularization $\lambda$"].astype(
            int)
        reg_align_df.append(align_df)
    df = pd.concat(reg_align_df)
    return df


def plot_relu(df_relu):
    align_relu_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                                   hue=r"Regularization $\lambda$", data=df_relu, legend="full")
    align_relu_fig = align_relu_plot.get_figure()
    align_relu_fig.savefig('align_relu_fig.pdf')


def plot_autograd_relu(df_relu):
    align_relu_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                                   hue=r"Regularization $\lambda$", data=df_relu, legend="full")
    align_relu_fig = align_relu_plot.get_figure()
    align_relu_fig.savefig('align_autograd_relu_fig.pdf')


def plot_lr(df_lr):
    align_lr_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                                   hue=r"Regularization $\lambda$", data=df_lr, legend="full")
    align_lr_fig = align_lr_plot.get_figure()
    align_lr_fig.savefig('align_lr_fig.pdf')


def plot_autograd_lr(df_relu):
    align_relu_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                                   hue=r"Regularization $\lambda$", data=df_relu, legend="full")
    align_relu_fig = align_relu_plot.get_figure()
    align_relu_fig.savefig('align_autograd_lr_fig.pdf')


#  # Generate alignment plot for relu network and nn data
#  n, d = (50, 150)
#  step = 10e-6
#  n_step = 6000
#  n_iter = 20
#  p_start = 300
#  p_end = 1000
#  p_step = 25
#  p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
#  reg_list = [0, 10, 20]
#  df_relu = get_align_df(n, d, p_list, reg_list, 'relu',
#                        'nn', step, n_step, n_iter)
#  plot_relu(df_relu)

#  # Generate alignment plot for linear network and lr data
#  n, d = (50, 150)
#  step = 10e-5
#  n_step = 2000
#  n_iter = 10
#  p_start = 300
#  p_end = 1000
#  p_step = 25
#  p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
#  reg_list = [0, 5, 10]
#  df_lr = get_align_df(n, d, p_list, reg_list, 'non', 'lr', step, n_step, n_iter)
#  plot_lr(df_lr)


# # Generate alignment plot for autograd relu network and nn data
# n, d = (50, 150)
# step = 10e-4
# n_step = 5000
# n_iter = 20
# p_start = 300
# p_end = 1000
# p_step = 25
# p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
# reg_list = [0, 2, 5]
# df_relu = get_autograd_align_df(n, d, p_list, reg_list, 'relu', 'nn', step, n_step, n_iter)
# plot_autograd_relu(df_relu)


# Generate alignment plot for autograd linear network and lr data
n, d = (50, 150)
step = 10e-4
n_step = 5000
n_iter = 20
p_start = 300
p_end = 1000
p_step = 25
p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
reg_list = [0, 2, 5]
df_lr = get_autograd_align_df(n, d, p_list, reg_list, 'non', 'lr', step, n_step, n_iter)
plot_autograd_lr(df_lr)
