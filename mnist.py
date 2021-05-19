import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import fa_autograd
import torchvision
from torchvision import datasets, transforms
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
    for batch_idx, (data, target) in enumerate(train_loader):
        pass


def train_epoch(torch_net_fa):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    len(mnist_trainset)
    train_loader = torch.utils.data.DataLoader(mnist_trainset)
    optimizer_fa = torch.optim.SGD(torch_net_fa.parameters(), lr=10e-6)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.flatten().unsqueeze(0).to(device)
        target = target.to(device)
        output = torch_net_fa(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer_fa.step()
        if batch_idx % 10000 == 9999:
            print(batch_idx, loss.item())


n_epochs = 10
torch_net_fa = MNISTTwoLayerFeedbackAlignmentNetworkReLU(1000, 0).to(device)
for epoch in range(n_epochs):
    train_epoch(torch_net_fa)
