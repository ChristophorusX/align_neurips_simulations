import fa_two_layer
import data_gen
import net_autograd
import simulation_align
import numpy as np
import torch
import argparse
import seaborn as sns
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 28})
# plt.rcParams["figure.figsize"] = (9, 9)
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# seed0 = np.random.randint(100000)
# seed1 = np.random.randint(100000)
# seed2 = np.random.randint(100000)
# n, d, p = (50, 10, 1000)
# # X, y = data_gen.lr_data(n, d)
# X, y = data_gen.rand_nn_data(n, d, p)
# step = 10e-5
# n_step = 10000
# activation = 'relu'
#
# net = fa_two_layer.TwoLayerNetwork(activation, d, p, n, seed1)
# loss_bp = net.back_propagation(X, y, step, n_step)
# loss_fa1, _, _ = net.feedback_alignment(
#     X, y, step, regular='non', n_steps=n_step)
# loss_fa10, _, _ = net.feedback_alignment(
#     X, y, step, regular=10, n_steps=n_step)
# plt.plot(np.arange(n_step)[:10000], loss_bp[:10000], np.arange(n_step)[
#          :10000], loss_fa1[:10000], np.arange(n_step)[:10000], loss_fa10[:10000])
#
#
# learning_rate = 10e-4
# reg = 0.1
# torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkReLU(
#     d, p, reg)
# optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=learning_rate)
# loss_fn_fa = nn.MSELoss()
# X_torch = torch.FloatTensor(X)
# y_torch = torch.FloatTensor(y).unsqueeze(1)
# loss_fa_autograd = []
# for t in range(n_step):
#     pred = torch_net_fa.forward(X_torch)
#     loss = loss_fn_fa(pred, y_torch)
#     loss_fa_autograd.append(loss.item())
#     if t % 100 == 99:
#         print(t, loss.item())
#     optimizer_fa.zero_grad()
#     loss.backward()
#     optimizer_fa.step()
#
# plt.plot(np.arange(n_step)[:10000], loss_fa_autograd[:10000])
#
#
# torch_net_bp = net_autograd.TwoLayerBackPropNetworkReLU(d, p)
# optimizer_bp = torch.optim.SGD(torch_net_bp.parameters(), lr=learning_rate)
# loss_fn_bp = nn.MSELoss()
# y_torch = torch.FloatTensor(y).unsqueeze(1)
# y_torch.size()
# loss_bp_autograd = []
# for t in range(n_step):
#     X = torch.FloatTensor(X)
#     pred = torch_net_bp.forward(X)
#     loss = loss_fn_bp(pred, y_torch)
#     loss_bp_autograd.append(loss.item())
#     if t % 100 == 99:
#         print(t, loss.item())
#     optimizer_bp.zero_grad()
#     loss.backward()
#     optimizer_bp.step()
#
# plt.plot(np.arange(n_step)[:1000], loss_bp_autograd[:1000], np.arange(
#     n_step)[:1000], loss_fa_autograd[:1000])
#
#
# drop_rate = 0.1
# torch_net_dropout = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkReLU(
#     d, p, drop_rate)
# optimizer_dropout = torch.optim.SGD(
#     torch_net_dropout.parameters(), lr=learning_rate)
# loss_fn_dropout = nn.MSELoss()
# y_torch = torch.FloatTensor(y).unsqueeze(1)
# y_torch.size()
# loss_dropout_autograd = []
# for t in range(n_step):
#     X = torch.FloatTensor(X)
#     pred = torch_net_dropout.forward(X)
#     loss = loss_fn_dropout(pred, y_torch)
#     loss_dropout_autograd.append(loss.item())
#     if t % 100 == 99:
#         print(t, loss.item())
#     optimizer_dropout.zero_grad()
#     loss.backward()
#     optimizer_dropout.step()
#
# plt.plot(np.arange(n_step)[:10000], loss_dropout_autograd[:10000], np.arange(
#     n_step)[:10000], loss_fa_autograd[:10000])


def get_autograd_loss_df(n, d, p, reg_list, activation, synthetic_data, step, n_step, reg_step, n_iter, dropout=False):
    reg_loss_df = []
    for reg in reg_list:
        loss_table = []
        for iter in range(n_iter):
            loss_array = []
            if synthetic_data == 'lr':
                X, y = data_gen.lr_data(n, d)
            elif synthetic_data == 'nn':
                X, y = data_gen.rand_nn_data(n, d, p, activation)
            torch_net_fa = simulation_align.get_network(
                d, p, activation, reg, dropout)
            torch_net_fa.to(device)
            optimizer_fa = torch.optim.SGD(
                torch_net_fa.parameters(), lr=step)
            loss_fn_fa = nn.MSELoss()
            X_torch = torch.FloatTensor(X).to(device)
            y_torch = torch.FloatTensor(y).unsqueeze(1).to(device)
            if reg == 0:
                reg_flag = False
            else:
                if reg_step == 0:
                    reg_flag = False
                else:
                    reg_flag = True
            proportion_step = n_step
            continue_flag = False  # disable the continue flag
            t = 0
            align_record = 1
            loss_record = 0
            while t < proportion_step or continue_flag:
                if reg_flag is True and t >= reg_step - 1:
                    print("Stop regularization at step {}".format(t))
                    reg_flag = False
                    if dropout:
                        torch_net_fa.drop = nn.Dropout(0)
                    else:
                        for name, param in torch_net_fa.named_parameters():
                            if name == 'second_layer.regularization':
                                param.data.copy_(torch.zeros_like(param.data))
                pred = torch_net_fa.forward(X_torch)
                loss = loss_fn_fa(pred, y_torch)
                if t % (n_step / 5) == n_step / 5 - 1:
                    for name, param in torch_net_fa.named_parameters():
                        if name == 'second_layer.backprop_weight':
                            backprop_weight = param.data
                        if name == 'second_layer.weight':
                            second_layer_weight = param.data
                    align = torch.tensordot(backprop_weight, second_layer_weight) / \
                        torch.norm(backprop_weight) / \
                        torch.norm(second_layer_weight)
                    align = align.cpu().data.detach().numpy().flatten()
                    print(t, loss.item(), align)
                    if dropout:
                        if np.abs(align - align_record) < 0.0001:
                            continue_flag = False
                        align_record = align
                    else:
                        if np.abs(loss.item() - loss_record) < 0.001:
                            continue_flag = False
                        loss_record = loss.item()
                optimizer_fa.zero_grad()
                loss_array.append(loss.item())
                loss.backward()
                optimizer_fa.step()
                t += 1
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
            loss_table.append(loss_array)
        loss_table = np.array(loss_table)
        flattened_table = loss_table.flatten()
        step_index = np.repeat(np.arange(n_step), n_iter)
        reg_index = np.repeat(reg, n_step * n_iter)
        activation_index = np.repeat(activation, n_step * n_iter)
        combined_table = np.vstack(
            (step_index, flattened_table, reg_index, activation_index)).T
        loss_df = pd.DataFrame(data=combined_table, columns=[
                                'Step', 'Loss', r"Regularization $\lambda$", 'Activation'])
        loss_df['Step'] = loss_df['Step'].astype(int)
        loss_df['Loss'] = loss_df['Loss'].astype(np.double)
        loss_df[r"Regularization $\lambda$"] = loss_df[r"Regularization $\lambda$"].astype(
            float)
        reg_loss_df.append(loss_df)
    df = pd.concat(reg_loss_df)
    return df


def plot_loss(df, filename, n_category=4, manual_legend=False):
    custom_palette = sns.color_palette("CMRmap_r", n_category)
    sns.set(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    align_plot = sns.lineplot(x='Step', y='Loss',
                              hue=r"Regularization $\lambda$",
                              ci='sd', data=df, legend="full",
                              palette=custom_palette, ax=ax, linestyle='--')
    ax.set_yscale('log')
    if manual_legend:
        plt.legend(loc=(0.65, 0.1), title=r"Regularization $\lambda$")
    ax.set_xlabel('Step', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    align_fig = align_plot.get_figure()
    align_fig.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Loss Simulations")
    parser.add_argument('-d', '--data')
    parser.add_argument('-n', '--network')
    parser.add_argument('-s', '--scheme')
    parser.add_argument('-r', '--regularization')

    args = parser.parse_args()

    # Generate loss plot for autograd relu network and nn data
    if args.data == 'nn' and args.network == 'relu' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate loss plot for autograd relu network and nn data")
        n, d = (50, 150)
        step = 10e-3
        n_step = 5000
        reg_step = 0
        n_iter = 10
        p = 3200
        reg_list = [0, 0.02, 0.05, 0.2]
        df_relu = get_autograd_loss_df(
            n, d, p, reg_list, 'relu', 'nn', step, n_step, reg_step, n_iter)
        plot_loss(df_relu, "outputs/loss_{}_{}_{}_{}_v1.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_relu.to_csv("dataframes/df_loss_{}_{}_{}_{}_v1.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate loss plot for autograd sigmoid network and nn data
    if args.data == 'nn' and args.network == 'sigmoid' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate loss plot for autograd sigmoid network and nn data")
        n, d = (50, 150)
        step = 10e-2
        n_step = 5000
        reg_step = 0
        n_iter = 10
        p = 3200
        reg_list = [0, 0.001, 0.003, 0.01]
        df_sigmoid = get_autograd_loss_df(
            n, d, p, reg_list, 'sigmoid', 'nn', step, n_step, reg_step, n_iter)
        plot_loss(df_sigmoid, "outputs/loss_{}_{}_{}_{}_v1.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_sigmoid.to_csv("dataframes/df_loss_{}_{}_{}_{}_v1.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate loss plot for autograd tanh network and nn data
    if args.data == 'nn' and args.network == 'tanh' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate loss plot for autograd tanh network and nn data")
        n, d = (50, 150)
        step = 10e-2
        n_step = 5000
        reg_step = 0
        n_iter = 10
        p = 3200
        reg_list = [0, 0.001, 0.002, 0.005]
        df_sigmoid = get_autograd_loss_df(
            n, d, p, reg_list, 'tanh', 'nn', step, n_step, reg_step, n_iter)
        plot_loss(df_sigmoid, "outputs/loss_{}_{}_{}_{}_v1.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_sigmoid.to_csv("dataframes/df_loss_{}_{}_{}_{}_v1.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate loss plot for autograd linear network and lr data
    if args.data == 'lr' and args.network == 'non' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate loss plot for autograd linear network and lr data")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        reg_step = 0
        n_iter = 10
        p = 3200
        reg_list = [0, 0.2, 0.3, 0.5]
        df_lr = get_autograd_loss_df(
            n, d, p, reg_list, 'non', 'lr', step, n_step, reg_step, n_iter)
        plot_loss(df_lr, "outputs/loss_{}_{}_{}_{}_v1.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_lr.to_csv("dataframes/df_loss_{}_{}_{}_{}_v1.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)
