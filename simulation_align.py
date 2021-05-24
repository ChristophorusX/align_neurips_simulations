import fa_two_layer
import net_autograd
import data_gen
import numpy as np
import torch
from torch import nn
import pandas as pd
import seaborn as sns
import argparse
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
                    X, y = data_gen.rand_nn_data(n, d, p, activation)
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


def get_autograd_align_df(n, d, p_list, reg_list, activation, synthetic_data, step, n_step, n_iter, drop_out=False):
    reg_align_df = []
    for reg in reg_list:
        align_table = []
        for p in p_list:
            align_array = []
            for iter in range(n_iter):
                if synthetic_data == 'lr':
                    X, y = data_gen.lr_data(n, d)
                elif synthetic_data == 'nn':
                    X, y = data_gen.rand_nn_data(n, d, p, activation)
                if activation == 'relu':
                    if drop_out:
                        torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkReLU(
                            d, p, reg)
                    else:
                        torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkReLU(
                            d, p, reg)
                elif activation == 'sigmoid':
                    torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkSigmoid(
                        d, p, reg)
                elif activation == 'non':
                    torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkLinear(
                        d, p, reg)
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
            float)
        reg_align_df.append(align_df)
    df = pd.concat(reg_align_df)
    return df


def plot_align(df, filename):
    custom_palette = sns.color_palette("CMRmap_r", 4)
    fig, ax = plt.subplots()
    ax.set(xscale="log")
    align_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                              hue=r"Regularization $\lambda$", err_style="bars",
                              ci='sd', data=df, legend="full", markers=True,
                              palette=custom_palette, ax=ax)
    align_fig = align_plot.get_figure()
    align_fig.savefig(filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Alignment Simulations")
    parser.add_argument('-d', '--data')
    parser.add_argument('-n', '--network')
    parser.add_argument('-s', '--scheme')
    parser.add_argument('-r', '--regularization')

    args = parser.parse_args()

    # Generate alignment plot for relu network and nn data
    if args.data == 'nn' and args.network == 'relu' and args.scheme == 'manual' and args.regularization == 'l2':
        print("Generate alignment plot for relu network and nn data")
        n, d = (50, 150)
        step = 10e-6
        n_step = 6000
        n_iter = 2
        p_start = 9000
        p_end = 10000
        p_step = 1000
        p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        reg_list = [0, 10, 20]
        df_relu = get_align_df(n, d, p_list, reg_list, 'relu',
                               'nn', step, n_step, n_iter)
        plot_align(df_relu, 'outputs/align_relu_fig_large.pdf')

    # Generate alignment plot for linear network and lr data
    if args.data == 'lr' and args.network == 'non' and args.scheme == 'manual' and args.regularization == 'l2':
        print("Generate alignment plot for linear network and lr data")
        n, d = (50, 150)
        step = 10e-5
        n_step = 2000
        n_iter = 10
        p_start = 300
        p_end = 1000
        p_step = 25
        p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        reg_list = [0, 5, 10]
        df_lr = get_align_df(n, d, p_list, reg_list, 'non',
                             'lr', step, n_step, n_iter)
        plot_align(df_lr, 'outputs/align_lr_fig_large.pdf')

    # Generate alignment plot for autograd relu network and nn data
    if args.data == 'nn' and args.network == 'relu' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd relu network and nn data")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        n_iter = 10
        p_start = 5000
        p_end = 10000
        p_step = 100
        p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        reg_list = [0, 1, 2]
        df_relu = get_autograd_align_df(
            n, d, p_list, reg_list, 'relu', 'nn', step, n_step, n_iter)
        plot_align(df_relu, 'outputs/align_autograd_relu_fig_large.pdf')
        df_relu.to_csv('dataframes/df_relu_large.csv', index=False)

    # Generate alignment plot for autograd sigmoid network and nn data
    if args.data == 'nn' and args.network == 'sigmoid' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd sigmoid network and nn data")
        n, d = (50, 150)
        step = 10e-2
        n_step = 5000
        n_iter = 10
        p_start = 5000
        p_end = 6000
        p_step = 1000
        p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        reg_list = [0, 0.01, 0.1]
        df_sigmoid = get_autograd_align_df(
            n, d, p_list, reg_list, 'sigmoid', 'nn', step, n_step, n_iter)
        plot_align(df_sigmoid, 'outputs/align_autograd_sigmoid_fig_large.pdf')
        df_sigmoid.to_csv('dataframes/df_sigmoid_large.csv', index=False)

    # Generate alignment plot for autograd linear network and lr data
    if args.data == 'lr' and args.network == 'non' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd linear network and lr data")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        n_iter = 3
        # p_start = 5000
        # p_end = 10000
        # p_step = 1000
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [300, 600, 1200]
        reg_list = [0, 0.5, 1, 1.5]
        df_lr = get_autograd_align_df(
            n, d, p_list, reg_list, 'non', 'lr', step, n_step, n_iter)
        plot_align(df_lr, "outputs/align_{}_{}_{}_{}.pdf".format(args.data,
                   args.network, args.scheme, args.regularization))
        df_lr.to_csv("dataframes/df_{}_{}_{}_{}.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate alignment plot for autograd relu network and nn data with dropout
    if args.data == 'nn' and args.network == 'relu' and args.scheme == 'autograd' and args.regularization == 'dropout':
        print("Generate alignment plot for autograd relu network and nn data with dropout")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        n_iter = 10
        p_start = 5000
        p_end = 10000
        p_step = 100
        p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        reg_list = [0, 0.3, 0.6]
        df_relu = get_autograd_align_df(
            n, d, p_list, reg_list, 'relu', 'nn', step, n_step, n_iter, drop_out=True)
        plot_align(df_relu, 'outputs/align_autograd_relu_dropout_fig_large.pdf')
        df_relu.to_csv('dataframes/df_relu_dropout_large.csv', index=False)
