import torch
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


def get_align_mnist(torch_net_fa):
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
        error_signal = torch_net_fa.prediction.grad
        delta_fa = error_signal.mm(backprop_weight)
        delta_bp = error_signal.mm(second_layer_weight)
        align_vec = torch.tensordot(delta_fa, delta_bp) / \
            torch.norm(delta_fa) / torch.norm(delta_bp)
        align_vec = align_vec.cpu().data.detach().numpy().flatten()
        align_weight = torch.tensordot(backprop_weight, second_layer_weight) / \
            torch.norm(backprop_weight) / torch.norm(second_layer_weight)
        align_weight = align_weight.cpu().data.detach().numpy().flatten()
        return np.array([align_vec, align_weight]).flatten()


def train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, n_epochs, lr, batch_size, align_array, loss_array, accuracy_array, reg_type=None):
    reg_cnt = 0
    for epo in range(n_epochs):
        train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_testset)
        optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=lr)
        for batch_idx, (data, target) in enumerate(train_loader):
            # torch_net_fa.train()
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
                else: # cutoff reg
                    if reg_cnt == reg_type:
                        print("Reached regularization cutoff...")
                        for name, param in torch_net_fa.named_parameters():
                            if name == 'second_layer.regularization':
                                param.data.copy_(torch.zeros_like(param.data))
                            if name == 'third_layer.regularization':
                                param.data.copy_(torch.zeros_like(param.data))
            optimizer_fa.zero_grad()
            output = torch_net_fa(data)
            loss = F.nll_loss(output, target)
            torch_net_fa.prediction.retain_grad()
            for name, param in torch_net_fa.named_parameters():
                if name == 'third_layer.backprop_weight':
                    torch_net_fa.hidden2.retain_grad()
            loss.backward()
            optimizer_fa.step()
            align = get_align_mnist(torch_net_fa)
            align_array.append(align)
            reg_cnt = reg_cnt + 1
            if batch_idx % 100 == 99:
                print(align)
                # torch_net_fa.eval()
                test_loss = 0
                n_correct = 0
                with torch.no_grad():
                    for data_test, target_test in test_loader:
                        data_test = data_test.flatten().unsqueeze(0).to(device)
                        target_test = target_test.to(device)
                        output_test = torch_net_fa(data_test)
                        test_loss += F.nll_loss(output_test,
                                                target_test, reduction='sum').item()
                        pred_test = output_test.argmax(dim=1, keepdim=True)
                        n_correct += pred_test.eq(
                            target_test.view_as(pred_test)).sum().item()
                test_loss /= len(test_loader.dataset)
                accuracy = n_correct / len(test_loader.dataset)
                loss_array.append(test_loss)
                accuracy_array.append(accuracy)
                print(batch_idx, test_loss, accuracy)


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
        train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, n_epochs, lr,
                       batch_size, align_array, loss_array, accuracy_array)
        align_array = np.array(align_array)
        align_array
        reg_index = np.repeat(reg, align_array.shape[0])
        step_index = np.arange(align_array.shape[0]) * 1000
        combined_table = np.vstack((align_array.T, reg_index, step_index)).T
        if n_layers == 2:
            align_df = pd.DataFrame(data=combined_table, columns=[
                                    "Second Layer Vec Alignment", "Second Layer Weight Alignment", r"Regularization $\lambda$", "Step"])
        else:
            align_df = pd.DataFrame(data=combined_table, columns=["Second Layer Vec Alignment", 'Third Layer Vec Alignment',
                                    "Second Layer Weight Alignment", 'Third Layer Weight Alignment', r"Regularization $\lambda$", "Step"])
        reg_align_df.append(align_df)
        accuracy_array = np.array(accuracy_array)
        loss_array = np.array(loss_array)
        reg_index = np.repeat(reg, accuracy_array.shape[0])
        step_index = np.arange(accuracy_array.shape[0]) * 1000
        performance_table = np.vstack(
            (loss_array, accuracy_array, reg_index, step_index)).T
        performance_df = pd.DataFrame(data=performance_table, columns=[
                                      "Loss", "Accuracy", r"Regularization $\lambda$", "Step"])
        reg_performance_df.append(performance_df)
    align_df = pd.concat(reg_align_df)
    performance_df = pd.concat(reg_performance_df)
    return align_df, performance_df


def plot_mnist(align_df, performance_df, filename, n_category=4, n_layers=3):
    custom_palette = sns.color_palette("CMRmap_r", n_category)
    if n_layers == 2:
        fig = plt.figure(figsize=(8, 18))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        sns.lineplot(x='Step', y='Second Layer Vec Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ax=ax1)
        sns.lineplot(x='Step', y='Second Layer Weight Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ax=ax2)
        sns.lineplot(x='Step', y='Accuracy',
                     hue=r"Regularization $\lambda$", data=performance_df, legend="full",
                     palette=custom_palette, ax=ax3)
        fig.savefig(filename)
    else:
        fig = plt.figure(figsize=(8, 30))
        ax1 = plt.subplot(511)
        ax2 = plt.subplot(512)
        ax3 = plt.subplot(513)
        ax4 = plt.subplot(514)
        ax5 = plt.subplot(515)
        sns.lineplot(x='Step', y='Second Layer Vec Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ax=ax1)
        sns.lineplot(x='Step', y='Third Layer Vec Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ax=ax2)
        sns.lineplot(x='Step', y='Second Layer Weight Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ax=ax3)
        sns.lineplot(x='Step', y='Third Layer Weight Alignment',
                     hue=r"Regularization $\lambda$", data=align_df, legend="full",
                     palette=custom_palette, ax=ax4)
        sns.lineplot(x='Step', y='Accuracy',
                     hue=r"Regularization $\lambda$", data=performance_df, legend="full",
                     palette=custom_palette, ax=ax5)
        fig.savefig(filename)


if __name__ == '__main__':
    n_hidden = 1000
    lr = 1e-2
    n_epochs = 50
    batch_size = 600
    n_layers = 2
    reg_levels = [0, 0.1, 0.5]
    align_df, performance_df = get_mnist_align_df(
        n_epochs, n_hidden, lr, batch_size, reg_levels, n_layers=n_layers)
    align_df.to_csv(
        "dataframes/df_mnist_align_{}l.csv".format(n_layers), index=False)
    performance_df.to_csv(
        "dataframes/df_mnist_performance_{}l.csv".format(n_layers), index=False)
    plot_mnist(align_df, performance_df,
               "outputs/mnist_{}l.pdf".format(n_layers), len(reg_levels), n_layers=n_layers)
