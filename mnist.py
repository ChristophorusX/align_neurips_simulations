import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import fa_autograd
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
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

        self.first_layer = fa_autograd.FeedbackAlignmentReLU(
            self.input_features, self.hidden_features)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 10, regularization)

    def forward(self, X):
        hidden = self.first_layer(X)
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return F.log_softmax(prediction, dim=1)


def get_align_mnist():
    pass


def train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, align_array, loss_array, accuracy_array):
    train_loader = torch.utils.data.DataLoader(mnist_trainset)
    test_loader = torch.utils.data.DataLoader(mnist_testset)
    optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=10e-6)
    for batch_idx, (data, target) in enumerate(train_loader):
        torch_net_fa.train()
        data = data.flatten().unsqueeze(0).to(device)
        target = target.to(device)
        output = torch_net_fa(data)
        loss = F.nll_loss(output, target)
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
        align_array.append(align)
        if batch_idx % 1000 == 999:
            print(align)
            torch_net_fa.eval()
            test_loss = 0
            n_correct = 0
            with torch.no_grad():
                for data_test, target_test in test_loader:
                    data_test = data_test.flatten().unsqueeze(0).to(device)
                    target_test = target_test.to(device)
                    output_test = torch_net_fa(data_test)
                    test_loss += F.nll_loss(output_test, target_test, reduction='sum').item()
                    pred_test = output_test.argmax(dim=1, keepdim=True)
                    n_correct += pred_test.eq(target_test.view_as(pred_test)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy = n_correct / len(test_loader.dataset)
            loss_array.append(test_loss)
            accuracy_array.append(accuracy)
            print(batch_idx, test_loss, accuracy)


n_epochs = 1
align_array = []
loss_array = []
accuracy_array = []
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
torch_net_fa = MNISTTwoLayerFeedbackAlignmentNetworkReLU(1000, 0).to(device)
for epoch in range(n_epochs):
    train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, align_array, loss_array, accuracy_array)
align_plot = plt.plot(np.arange(n_epochs * len(mnist_trainset)), align_array, label='Two Layer ReLU')
plt.xlabel('# of samples')
plt.ylabel('Alignment')
plt.savefig('mnist_relu_alignment.pdf')
