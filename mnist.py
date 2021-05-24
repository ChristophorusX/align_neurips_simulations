import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
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
        self.prediction = Variable(torch.FloatTensor(1, 10), requires_grad=True)

    def forward(self, X):
        hidden=self.first_layer(X)
        # self.prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        self.prediction=self.second_layer(
            hidden) / np.sqrt(self.hidden_features)
        return F.log_softmax(self.prediction, dim=1)


class MNISTThreeLayerFeedbackAlignmentNetworkReLU(nn.Module):
    def __init__(self, hidden_features, regularization):
        super(MNISTThreeLayerFeedbackAlignmentNetworkReLU, self).__init__()
        self.input_features=784
        self.hidden_features=hidden_features

        self.first_layer=fa_autograd.FeedbackAlignmentReLU(
            self.input_features, self.hidden_features)
        self.second_layer=fa_autograd.FeedbackAlignmentReLU(
            self.hidden_features, self.hidden_features)
        self.third_layer=fa_autograd.RegLinear(
            self.hidden_features, 10, regularization)

    def forward(self, X):
        hidden=self.first_layer(X)
        hidden2=self.second_layer(hidden)
        prediction=self.second_layer(hidden2) / self.hidden_features
        return F.log_softmax(prediction, dim=1)


def get_align_mnist():
    pass


def train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset, align_array, loss_array, accuracy_array):
    train_loader=torch.utils.data.DataLoader(mnist_trainset)
    test_loader=torch.utils.data.DataLoader(mnist_testset)
    optimizer_fa=torch.optim.SGD(torch_net_fa.parameters(), lr=10e-4)
    for batch_idx, (data, target) in enumerate(train_loader):
        # torch_net_fa.train()
        data=data.flatten().unsqueeze(0).to(device)
        target=target.to(device)
        optimizer_fa.zero_grad()
        output=torch_net_fa(data)
        loss=F.nll_loss(output, target)
        print(torch_net_fa.prediction.grad)
        torch_net_fa.prediction.retain_grad()
        loss.backward()
        print(torch_net_fa.prediction.grad)
        optimizer_fa.step()
        for name, param in torch_net_fa.named_parameters():
            if name == 'second_layer.backprop_weight':
                backprop_weight=param.data
            if name == 'second_layer.weight':
                second_layer_weight=param.data
        error_signal=torch_net_fa.prediction.grad
        delta_fa=error_signal.mm(backprop_weight)
        delta_bp=error_signal.mm(second_layer_weight)
        align=torch.tensordot(delta_fa, delta_bp) / \
            torch.norm(delta_fa) / torch.norm(delta_bp)
        align=align.cpu().data.detach().numpy().flatten()
        align_array.append(align)
        if batch_idx % 1000 == 999:
            print(align)
            # torch_net_fa.eval()
            test_loss=0
            n_correct=0
            with torch.no_grad():
                for data_test, target_test in test_loader:
                    data_test=data_test.flatten().unsqueeze(0).to(device)
                    target_test=target_test.to(device)
                    output_test=torch_net_fa(data_test)
                    test_loss += F.nll_loss(output_test,
                                            target_test, reduction='sum').item()
                    pred_test=output_test.argmax(dim=1, keepdim=True)
                    n_correct += pred_test.eq(
                        target_test.view_as(pred_test)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy=n_correct / len(test_loader.dataset)
            loss_array.append(test_loss)
            accuracy_array.append(accuracy)
            print(batch_idx, test_loss, accuracy)


n_epochs=1
transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_trainset=datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
mnist_testset=datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
torch_net_fa=MNISTTwoLayerFeedbackAlignmentNetworkReLU(1000, 0).to(device)
torch_net_fa_reg=type(torch_net_fa)(1000, 0).to(device)
torch_net_fa_reg.load_state_dict(torch_net_fa.state_dict())
for name, param in torch_net_fa_reg.named_parameters():
    if name == 'second_layer.regularization':
        print('modifying regularization...')
        param.data.copy_(0.1 * torch.ones_like(param.data))
align_array=[]
loss_array=[]
accuracy_array=[]
for epoch in range(n_epochs):
    train_epoch_fa(torch_net_fa, mnist_trainset, mnist_testset,
                   align_array, loss_array, accuracy_array)
# align_plot = plt.plot(np.arange(n_epochs * len(mnist_trainset)), align_array, label='Two Layer ReLU')
# accuracy_plot = plt.plot(np.arange(len(accuracy_array)), accuracy_array)
align_array_reg=[]
loss_array_reg=[]
accuracy_array_reg=[]
for epoch in range(n_epochs):
    train_epoch_fa(torch_net_fa_reg, mnist_trainset, mnist_testset,
                   align_array_reg, loss_array_reg, accuracy_array_reg)
fig=plt.figure()
ax1=plt.subplot(211)
ax1.plot(np.arange(n_epochs * len(mnist_trainset)), align_array,
         np.arange(n_epochs * len(mnist_trainset)), align_array_reg)
ax2=plt.subplot(212)
ax2.plot(np.arange(len(accuracy_array)), accuracy_array,
         np.arange(len(accuracy_array_reg)), accuracy_array_reg)
fig.savefig('mnist_alignment_relu.pdf')
align_plot = plt.plot(np.arange(n_epochs * len(mnist_trainset)), align_array, np.arange(n_epochs * len(mnist_trainset)), align_array_reg)
accuracy_plot = plt.plot(np.arange(len(accuracy_array)), accuracy_array, np.arange(len(accuracy_array_reg)), accuracy_array_reg)
# plt.xlabel('# of samples')
# plt.ylabel('Alignment')
# plt.savefig('mnist_relu_alignment.pdf')
