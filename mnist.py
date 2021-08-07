import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import fa_autograd
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 28})
# plt.rcParams["figure.figsize"] = (9, 9)
# rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class MNISTTwoLayerFeedbackAlignmentNetworkReLU(nn.Module):
    def __init__(self, hidden_features, regularization):
        super(MNISTTwoLayerFeedbackAlignmentNetworkReLU, self).__init__()
        self.input_features = 784
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentRegReLU(
            self.input_features, self.hidden_features, regularization)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 10, regularization)
        self.prediction = Variable(
            torch.FloatTensor(1, 10), requires_grad=True)

    def forward(self, X):
        hidden = self.first_layer(X)
        # self.prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        self.prediction = self.second_layer(
            hidden) / np.sqrt(self.hidden_features)
        return F.log_softmax(self.prediction, dim=1)


class MNISTThreeLayerFeedbackAlignmentNetworkReLU(nn.Module):
    def __init__(self, hidden_features, regularization):
        super(MNISTThreeLayerFeedbackAlignmentNetworkReLU, self).__init__()
        self.input_features = 784
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentRegReLU(
            self.input_features, self.hidden_features, regularization)
        self.second_layer = fa_autograd.FeedbackAlignmentRegReLU(
            self.hidden_features, self.hidden_features, regularization)
        self.third_layer = fa_autograd.RegLinear(
            self.hidden_features, 10, regularization)
        self.hidden2 = Variable(torch.FloatTensor(
            1, self.hidden_features), requires_grad=True)
        self.prediction = Variable(
            torch.FloatTensor(1, 10), requires_grad=True)

    def forward(self, X):
        hidden = self.first_layer(X)
        self.hidden2 = self.second_layer(hidden)
        self.prediction = self.third_layer(self.hidden2) / self.hidden_features
        return F.log_softmax(self.prediction, dim=1)


def get_align_mnist(torch_net_fa, t: int, init_second_layer_weight, lr, reg):
    is_three_layer = False
    for name, param in torch_net_fa.named_parameters():
        if name == 'third_layer.backprop_weight':
            is_three_layer = True
    if is_three_layer:
        for name, param in torch_net_fa.named_parameters():
            if name == 'second_layer.backprop_weight':
                second_backprop_weight = param.data
            if name == 'second_layer.weight':
                second_layer_weight = param.data
            if name == 'third_layer.backprop_weight':
                third_backprop_weight = param.data
            if name == 'third_layer.weight':
                third_layer_weight = param.data
        second_error_signal = torch_net_fa.hidden2.grad
        second_delta_fa = second_error_signal.mm(second_backprop_weight)
        second_delta_bp = second_error_signal.mm(second_layer_weight)
        second_align_vec = torch.tensordot(second_delta_fa, second_delta_bp) / \
            torch.norm(second_delta_fa) / torch.norm(second_delta_bp)
        second_align_vec = second_align_vec.cpu().data.detach().numpy().flatten()
        second_align_weight = torch.tensordot(second_backprop_weight, second_layer_weight) / \
            torch.norm(second_backprop_weight) / \
            torch.norm(second_layer_weight)
        second_align_weight = second_align_weight.cpu().data.detach().numpy().flatten()
        third_error_signal = torch_net_fa.prediction.grad
        third_delta_fa = third_error_signal.mm(third_backprop_weight)
        third_delta_bp = third_error_signal.mm(third_layer_weight)
        third_align_vec = torch.tensordot(third_delta_fa, third_delta_bp) / \
            torch.norm(third_delta_fa) / torch.norm(third_delta_bp)
        third_align_vec = third_align_vec.cpu().data.detach().numpy().flatten()
        third_align_weight = torch.tensordot(third_backprop_weight, third_layer_weight) / \
            torch.norm(third_backprop_weight) / torch.norm(third_layer_weight)
        third_align_weight = third_align_weight.cpu().data.detach().numpy().flatten()
        return np.array([third_align_vec, third_align_vec, third_align_weight, third_align_weight]).flatten()
    else:
        for name, param in torch_net_fa.named_parameters():
            if name == 'second_layer.backprop_weight':
                backprop_weight = param.data
            if name == 'second_layer.weight':
                second_layer_weight = param.data
        disentangled_weight = second_layer_weight - init_second_layer_weight + (1-0.1*lr)**t * init_second_layer_weight
        error_signal = torch_net_fa.prediction.grad
        delta_fa = error_signal.mm(backprop_weight)
        delta_disentangled = error_signal.mm(disentangled_weight)
        delta_bp = error_signal.mm(second_layer_weight)
        align_vec = 0
        for row in range(delta_fa.shape[0]):
            norm_fa = torch.norm(delta_fa[row])
            norm_bp = torch.norm(delta_bp[row])
            if norm_fa != 0 and norm_bp != 0:
                align_vec += torch.dot(delta_fa[row], delta_bp[row]) / \
                    norm_fa / norm_bp
        align_vec = align_vec / delta_fa.shape[0]
        align_vec = align_vec.cpu().data.detach().numpy().flatten()
        align_disentangled = 0
        for row in range(delta_disentangled.shape[0]):
            norm_disentangled = torch.norm(delta_disentangled[row])
            norm_fa = torch.norm(delta_fa[row])
            if norm_disentangled != 0 and norm_fa != 0:
                align_disentangled += torch.dot(delta_disentangled[row], delta_fa[row]) / \
                    norm_disentangled / norm_fa
        align_disentangled = align_disentangled / delta_disentangled.shape[0]
        align_disentangled = align_disentangled.cpu().data.detach().numpy().flatten()
        align_weight = torch.tensordot(backprop_weight, second_layer_weight) / \
            torch.norm(backprop_weight) / torch.norm(second_layer_weight)
        align_weight = align_weight.cpu().data.detach().numpy().flatten()
        return np.array([align_vec, align_weight, align_disentangled]).flatten(), disentangled_weight, second_layer_weight, backprop_weight


def train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, n_epochs, lr, batch_size, align_array, loss_array, accuracy_array, weights_array, disentangled_weights_array, backprop_weights_array, reg_type=None):
    reg_cnt = 0
    t = 0
    for name, param in torch_net_fa.named_parameters():
        if name == 'second_layer.weight':
            init_second_layer_weight = param.data.clone()
    for epo in range(n_epochs):
        train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=500)
        optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=lr)
        for batch_idx, (data, target) in enumerate(train_loader):
            # torch_net_fa.train()
            t += 1
            data = data.flatten(start_dim=1).to(device)
            target = target.to(device)
            if reg_type is not None:
                if reg_type == 'exp':
                    for name, param in torch_net_fa.named_parameters():
                        if name == 'second_layer.regularization':
                            # param.data.copy_(reg * torch.ones_like(param.data))
                            param.data.copy_(0.9999999999999 * param.data)
                        if name == 'third_layer.regularization':
                            # param.data.copy_(reg * torch.ones_like(param.data))
                            param.data.copy_(0.9999999999999 * param.data)
                elif reg_type == 'cutoff': # cutoff reg
                    if reg_cnt == reg_type:
                        print("Reached regularization cutoff...")
                        for name, param in torch_net_fa.named_parameters():
                            if name == 'second_layer.regularization':
                                param.data.copy_(torch.zeros_like(param.data))
                            if name == 'third_layer.regularization':
                                param.data.copy_(torch.zeros_like(param.data))
                else:
                    reg = reg_type
            optimizer_fa.zero_grad()
            output = torch_net_fa(data)
            loss = F.nll_loss(output, target)
            torch_net_fa.prediction.retain_grad()
            for name, param in torch_net_fa.named_parameters():
                if name == 'third_layer.backprop_weight':
                    torch_net_fa.hidden2.retain_grad()
            loss.backward()
            optimizer_fa.step()
            reg_cnt = reg_cnt + 1
            if batch_idx % 100 == 99:
                align, disentangled_weight, second_layer_weight, backprop_weight = get_align_mnist(torch_net_fa, t, init_second_layer_weight, lr, reg)
                align_array.append(align)
                print(align)
                weights_array.append(second_layer_weight.numpy())
                disentangled_weights_array.append(disentangled_weight.numpy())
                backprop_weights_array.append(backprop_weight.numpy())
                # torch_net_fa.eval()
                test_loss = 0
                n_correct = 0
                with torch.no_grad():
                    for data_test, target_test in test_loader:
                        data_test = data_test.flatten(start_dim=1).to(device)
                        target_test = target_test.to(device)
                        output_test = torch_net_fa(data_test)
                        test_loss += F.nll_loss(output_test,
                                                target_test, reduction='sum').item()
                        pred_test = output_test.argmax(dim=1, keepdim=True)
                        n_correct += pred_test.eq(
                            target_test.view_as(pred_test)).sum().item()
                test_loss /= len(test_loader.dataset)
                accuracy = n_correct / len(test_loader.dataset)
                test_loss_disentangled = 0
                n_correct_disentangled = 0
                for name, param in torch_net_fa.named_parameters():
                    if name == 'second_layer.weight':
                        tmp = param.data.clone()
                        param.data.copy_(disentangled_weight)
                with torch.no_grad():
                    for data_test, target_test in test_loader:
                        data_test = data_test.flatten(start_dim=1).to(device)
                        target_test = target_test.to(device)
                        output_test = torch_net_fa(data_test)
                        test_loss_disentangled += F.nll_loss(output_test,
                                                target_test, reduction='sum').item()
                        pred_test = output_test.argmax(dim=1, keepdim=True)
                        n_correct_disentangled += pred_test.eq(
                            target_test.view_as(pred_test)).sum().item()
                for name, param in torch_net_fa.named_parameters():
                    if name == 'second_layer.weight':
                        param.data.copy_(tmp)
                test_loss_disentangled /= len(test_loader.dataset)
                accuracy_disentangled = n_correct_disentangled / len(test_loader.dataset)
                loss_array.append([test_loss, test_loss_disentangled])
                accuracy_array.append([accuracy, accuracy_disentangled])
                print(batch_idx, test_loss, accuracy, test_loss_disentangled, accuracy_disentangled)


def get_network_with_reg(torch_net_fa, n_hidden, reg):
    torch_net_fa_reg = type(torch_net_fa)(n_hidden, 0).to(device)
    torch_net_fa_reg.load_state_dict(torch_net_fa.state_dict())
    for name, param in torch_net_fa_reg.named_parameters():
        if name == 'second_layer.regularization':
            print('modifying second layer regularization...')
            param.data.copy_(reg * torch.ones_like(param.data))
        if name == 'third_layer.regularization':
            print('modifying third layer regularization...')
            param.data.copy_(reg * torch.ones_like(param.data))
    return torch_net_fa_reg


def get_mnist_align_df(n_epochs, n_hidden, lr, batch_size, reg_levels, n_layers=3):
    print("Preparing datasets...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    print("Preparing networks...")
    net_list = []
    if n_layers == 2:
        print("Fitting two-layer networks...")
        torch_net_fa0 = MNISTTwoLayerFeedbackAlignmentNetworkReLU(
            n_hidden, 0).to(device)
    else:
        print("Fitting three-layer networks...")
        torch_net_fa0 = MNISTThreeLayerFeedbackAlignmentNetworkReLU(
            n_hidden, 0).to(device)
    for reg in reg_levels:
        print("Generating network with reg level {}".format(reg))
        torch_net_fa_reg = get_network_with_reg(torch_net_fa0, n_hidden, reg)
        net_list.append(torch_net_fa_reg)
    print("Generating dataframes...")
    reg_align_df = []
    reg_performance_df = []
    zipped_list = zip(reg_levels, net_list)
    for reg, torch_net_fa in zipped_list:
        print("Working on regularization level {}".format(reg))
        align_array = []
        loss_array = []
        accuracy_array = []
        weights_array = []
        disentangled_weights_array = []
        backprop_weights_array = []
        train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, n_epochs, lr,
                       batch_size, align_array, loss_array, accuracy_array, weights_array,
                       disentangled_weights_array, backprop_weights_array, reg)
        align_array = np.array(align_array)
        align_array
        weights_array = np.array(weights_array)
        disentangled_weights_array = np.array(disentangled_weights_array)
        backprop_weights_array = np.array(backprop_weights_array)
        np.save("arrays/weights_reg{}.npy".format(reg), weights_array)
        np.save("arrays/disentangled_weights_reg{}.npy".format(reg), disentangled_weights_array)
        np.save("arrays/backprop_weights_reg{}.npy".format(reg), backprop_weights_array)
        reg_index = np.repeat(reg, align_array.shape[0])
        step_index = np.arange(align_array.shape[0]) * 100
        combined_table = np.vstack((align_array.T, reg_index, step_index)).T
        if n_layers == 2:
            align_df = pd.DataFrame(data=combined_table, columns=[
                                    "Second Layer Vec Alignment", "Second Layer Weight Alignment", "Disentangled Alignment", r"Regularization $\lambda$", "Step"])
        else:
            align_df = pd.DataFrame(data=combined_table, columns=["Second Layer Vec Alignment", 'Third Layer Vec Alignment',
                                    "Second Layer Weight Alignment", 'Third Layer Weight Alignment', r"Regularization $\lambda$", "Step"])
        reg_align_df.append(align_df)
        accuracy_array = np.array(accuracy_array)
        loss_array = np.array(loss_array)
        reg_index = np.repeat(reg, accuracy_array.shape[0])
        step_index = np.arange(accuracy_array.shape[0]) * 100
        performance_table = np.vstack(
            (loss_array.T, accuracy_array.T, reg_index, step_index)).T
        performance_df = pd.DataFrame(data=performance_table, columns=[
                                      "Loss", "Disentangled Loss", "Accuracy", "Disentangled Accuracy", r"Regularization $\lambda$", "Step"])
        reg_performance_df.append(performance_df)
    align_df = pd.concat(reg_align_df)
    performance_df = pd.concat(reg_performance_df)
    return align_df, performance_df


def plot_mnist(align_df, performance_df, filename, n_category=4, n_layers=3):
    custom_palette = sns.color_palette("CMRmap_r", n_category)
    sns.set(font_scale=1.1)
    if n_layers == 2:
        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        # ax2 = plt.subplot(132)
        ax3 = plt.subplot(122)
        align_df_reg0 = align_df.loc[align_df[r"Regularization $\lambda$"] == 0]
        performance_df_reg0 = performance_df.loc[performance_df[r"Regularization $\lambda$"] == 0]
        sns.lineplot(x='Step', y='Second Layer Vec Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax1, linestyle='-.')
        sns.lineplot(x='Step', y='Disentangled Alignment',
                     data=align_df_reg0, legend="full",
                     ci='sd', ax=ax1, linestyle=':')
        # sns.lineplot(x='Step', y='Second Layer Weight Alignment',
        #              hue=r"Regularization $\lambda$", data=align_df, legend="full",
        #              palette=custom_palette, ci='sd', ax=ax2, linestyle='-.')
        sns.lineplot(x='Step', y='Accuracy',
                     hue=r"Regularization $\lambda$", data=performance_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax3, linestyle='-.')
        sns.lineplot(x='Step', y='Disentangled Accuracy',
                     data=performance_df_reg0, legend="full",
                     ci='sd', ax=ax3, linestyle=':')
        ax1.set_xlabel('Step')
        # ax1.set_ylabel(r"$\frac{\langle \delta_{\mathrm{FA}},\delta_{\mathrm{BP}}\rangle}{\|\delta_{\mathrm{FA}}\|\|\delta_{\mathrm{BP}}\|}$", fontsize=18)
        ax1.set_ylabel('Alignment')
        # ax2.set_xlabel('Step', fontsize=18)
        # ax2.set_ylabel(r"$\frac{\langle \beta,b\rangle}{\|\beta\|\|b\|}$", fontsize=18)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Accuracy')
        fig.savefig(filename, bbox_inches='tight')
    else:
        fig = plt.figure(figsize=(8, 30))
        ax1 = plt.subplot(511)
        ax2 = plt.subplot(512)
        ax3 = plt.subplot(513)
        ax4 = plt.subplot(514)
        ax5 = plt.subplot(515)
        sns.lineplot(x='Step', y='Second Layer Vec Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax1)
        sns.lineplot(x='Step', y='Third Layer Vec Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax2)
        sns.lineplot(x='Step', y='Second Layer Weight Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax3)
        sns.lineplot(x='Step', y='Third Layer Weight Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax4)
        sns.lineplot(x='Step', y='Accuracy',
                     hue=r"Regularization $\lambda$", data=performance_df, legend="full",
                     palette=custom_palette, ci='sd', ax=ax5)
        fig.savefig(filename, bbox_inches='tight')


def load_df_arr(n_jobs):
    df_arr_performance = []
    df_arr_align = []
    for jobnumber in np.arange(n_jobs):
        df_performance = pd.read_csv("dataframes/df_mnist_performance_2l_v7_job{}.csv".format(jobnumber))
        df_arr_performance.append(df_performance)
        df_align = pd.read_csv("dataframes/df_mnist_align_2l_v7_job{}.csv".format(jobnumber))
        df_arr_align.append(df_align)
    return df_arr_align, df_arr_performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST Simulations")
    parser.add_argument('-j', '--jobnumber')
    args = parser.parse_args()
    print(args.jobnumber)
    
    n_hidden = 1000
    lr = 1e-2
    n_epochs = 3 # 300
    batch_size = 600
    n_layers = 2
    reg_levels = [0, 0.1, 0.3]
    align_df, performance_df = get_mnist_align_df(
        n_epochs, n_hidden, lr, batch_size, reg_levels, n_layers=n_layers)
    align_df.to_csv(
        "dataframes/df_mnist_align_{}l_v7_job{}.csv".format(n_layers, args.jobnumber), index=False)
    performance_df.to_csv(
        "dataframes/df_mnist_performance_{}l_v7_job{}.csv".format(n_layers, args.jobnumber), index=False)
    plot_mnist(align_df, performance_df,
               "outputs/mnist_{}l_v7_job{}.pdf".format(n_layers, args.jobnumber), len(reg_levels), n_layers=n_layers)

    # # Load df and redraw the figures
    # n_jobs = 1 #10
    # df_arr_align, df_arr_performance = load_df_arr(n_jobs)
    # align_df = pd.concat(df_arr_align)
    # align_df.shape
    # align_df['Step'] = align_df['Step'] // 1000
    # performance_df = pd.concat(df_arr_performance)
    # performance_df.shape
    # performance_df['Step'] = performance_df['Step'] // 10
    # align_subsampling = np.arange(align_df.shape[0], step=100)
    # a_df = align_df.iloc[align_subsampling, :]
    # plot_mnist(a_df, performance_df,
    #            "outputs/mnist_{}l_v7_horizontal.pdf".format(2), 3, 2)
