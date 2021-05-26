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
            for iter in np.arange(n_iter):
                if synthetic_data == 'lr':
                    X, y = data_gen.lr_data(n, d)
                elif synthetic_data == 'nn':
                    X, y = data_gen.rand_nn_data(n, d, p, activation)
                seed = np.random.randint(100000)
                net = fa_two_layer.TwoLayerNetwork(activation, d, p, n, seed)
                proportion_step = np.rint(n_step * np.sqrt(p) // np.rint(np.sqrt(p_list[0])))
                loss_fa, beta, b = net.feedback_alignment(
                    X, y, step, regular=reg, n_steps=proportion_step)
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


def get_network(d, p, activation, reg, dropout):
    if activation == 'relu':
        if dropout:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkReLU(
                d, p, reg)
            return torch_net_fa
        else:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkReLU(
                d, p, reg)
            return torch_net_fa
    elif activation == 'sigmoid':
        if dropout:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkSigmoid(
                d, p, reg)
            return torch_net_fa
        else:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkSigmoid(
                d, p, reg)
            return torch_net_fa
    elif activation == 'tanh':
        if dropout:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkTanh(
                d, p, reg)
            return torch_net_fa
        else:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkTanh(
                d, p, reg)
            return torch_net_fa
    elif activation == 'non':
        if dropout:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentDropoutNetworkLinear(
                d, p, reg)
            return torch_net_fa
        else:
            torch_net_fa = net_autograd.TwoLayerFeedbackAlignmentNetworkLinear(
                d, p, reg)
            return torch_net_fa


def get_autograd_align_df(n, d, p_list, reg_list, activation, synthetic_data, step, n_step, reg_step, n_iter, dropout=False):
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
                torch_net_fa = get_network(d, p, activation, reg, dropout)
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
                proportion_step = np.rint(n_step * np.sqrt(p) // np.rint(np.sqrt(p_list[0])))
                for t in np.arange(proportion_step):
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


def plot_align(df, filename, n_category=4):
    custom_palette = sns.color_palette("CMRmap_r", n_category)
    fig, ax = plt.subplots()
    ax.set(xscale="log")
    align_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                              hue=r"Regularization $\lambda$", err_style="bars",
                              ci='sd', data=df, legend="full", markers='o',
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
        n_iter = 3
        # p_start = 300
        # p_end = 1000
        # p_step = 25
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [500, 2000, 10000]
        reg_list = [0, 5, 10]
        df_lr = get_align_df(n, d, p_list, reg_list, 'non',
                             'lr', step, n_step, n_iter)
        plot_align(df_lr, 'outputs/align_lr_fig_large_v4.pdf', len(reg_list))

    # Generate alignment plot for autograd relu network and nn data
    if args.data == 'nn' and args.network == 'relu' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd relu network and nn data")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        reg_step = 0
        n_iter = 3
        # p_start = 5000
        # p_end = 10000
        # p_step = 100
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [1600, 6400, 12800]
        reg_list = [0, 0.5, 1, 2]
        df_relu = get_autograd_align_df(
            n, d, p_list, reg_list, 'relu', 'nn', step, n_step, reg_step, n_iter)
        plot_align(df_relu, "outputs/align_{}_{}_{}_{}_v2.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_relu.to_csv("dataframes/df_{}_{}_{}_{}_v2.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate alignment plot for autograd sigmoid network and nn data
    if args.data == 'nn' and args.network == 'sigmoid' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd sigmoid network and nn data")
        n, d = (50, 150)
        step = 10e-2
        n_step = 5000
        reg_step = 0
        n_iter = 3
        # p_start = 5000
        # p_end = 6000
        # p_step = 1000
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [200, 2400, 12800]
        reg_list = [0, 0.003, 0.005, 0.01]
        df_sigmoid = get_autograd_align_df(
            n, d, p_list, reg_list, 'sigmoid', 'nn', step, n_step, reg_step, n_iter)
        plot_align(df_sigmoid, "outputs/align_{}_{}_{}_{}_v2.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_sigmoid.to_csv("dataframes/df_{}_{}_{}_{}_v2.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate alignment plot for autograd tanh network and nn data
    if args.data == 'nn' and args.network == 'tanh' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd tanh network and nn data")
        n, d = (50, 150)
        step = 10e-2
        n_step = 5000
        reg_step = 0
        n_iter = 3
        # p_start = 5000
        # p_end = 6000
        # p_step = 1000
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [200, 2400, 12800]
        reg_list = [0, 0.003, 0.005, 0.01]
        df_sigmoid = get_autograd_align_df(
            n, d, p_list, reg_list, 'tanh', 'nn', step, n_step, reg_step, n_iter)
        plot_align(df_sigmoid, "outputs/align_{}_{}_{}_{}_v2.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_sigmoid.to_csv("dataframes/df_{}_{}_{}_{}_v2.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate alignment plot for autograd linear network and lr data
    if args.data == 'lr' and args.network == 'non' and args.scheme == 'autograd' and args.regularization == 'l2':
        print("Generate alignment plot for autograd linear network and lr data")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        reg_step = 0
        n_iter = 3
        # p_start = 5000
        # p_end = 10000
        # p_step = 1000
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [200, 2400, 12800]
        reg_list = [0, 0.5, 1, 1.5]
        df_lr = get_autograd_align_df(
            n, d, p_list, reg_list, 'non', 'lr', step, n_step, reg_step, n_iter)
        plot_align(df_lr, "outputs/align_{}_{}_{}_{}_v2.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_lr.to_csv("dataframes/df_{}_{}_{}_{}_v2.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)

    # Generate alignment plot for autograd relu network and nn data with dropout
    if args.data == 'nn' and args.network == 'relu' and args.scheme == 'autograd' and args.regularization == 'dropout':
        print("Generate alignment plot for autograd relu network and nn data with dropout")
        n, d = (50, 150)
        step = 10e-4
        n_step = 5000
        reg_step = 0
        n_iter = 15
        # p_start = 5000
        # p_end = 10000
        # p_step = 100
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [200, 400, 800, 1600, 3200, 6400, 12800]
        reg_list = [0, 0.4, 0.6, 0.8]
        df_relu = get_autograd_align_df(
            n, d, p_list, reg_list, 'relu', 'nn', step, n_step, reg_step, n_iter, dropout=True)
        plot_align(df_relu, "outputs/align_{}_{}_{}_{}_v2.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_relu.to_csv("dataframes/df_{}_{}_{}_{}_v2.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)


    # Generate alignment plot for autograd sigmoid network and nn data with dropout
    if args.data == 'nn' and args.network == 'sigmoid' and args.scheme == 'autograd' and args.regularization == 'dropout':
        print("Generate alignment plot for autograd sigmoid network and nn data with dropout")
        n, d = (50, 150)
        step = 10e-2
        n_step = 5000
        reg_step = 2000
        n_iter = 15
        # p_start = 5000
        # p_end = 10000
        # p_step = 100
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [200, 400, 800, 1600, 3200, 6400, 12800]
        reg_list = [0, 0.5, 0.7, 0.9]
        df_sigmoid = get_autograd_align_df(
            n, d, p_list, reg_list, 'sigmoid', 'nn', step, n_step, reg_step, n_iter, dropout=True)
        plot_align(df_sigmoid, "outputs/align_{}_{}_{}_{}.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_sigmoid.to_csv("dataframes/df_{}_{}_{}_{}.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)


    # Generate alignment plot for autograd linear network and lr data with dropout
    if args.data == 'lr' and args.network == 'non' and args.scheme == 'autograd' and args.regularization == 'dropout':
        print("Generate alignment plot for autograd linear network and lr data with dropout")
        n, d = (50, 150)
        step = 10e-4
        n_step = 10000
        reg_step = 0
        n_iter = 2
        # p_start = 5000
        # p_end = 10000
        # p_step = 100
        # p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
        p_list = [200, 2400, 12800]
        reg_list = [0, 0.4, 0.6, 0.8]
        df_lr = get_autograd_align_df(
            n, d, p_list, reg_list, 'non', 'lr', step, n_step, reg_step, n_iter, dropout=True)
        plot_align(df_lr, "outputs/align_{}_{}_{}_{}_v2.pdf".format(args.data,
                   args.network, args.scheme, args.regularization), len(reg_list))
        df_lr.to_csv("dataframes/df_{}_{}_{}_{}_v2.csv".format(args.data,
                     args.network, args.scheme, args.regularization), index=False)
