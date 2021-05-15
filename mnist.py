import torch
from torch import nn
import torch.nn.functional as F
import linear_torch


class LinearNetwork(nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list):
        """
        :param in_features: dimension of input features (784 for MNIST)
        :param num_layers: number of layers for feed-forward net
        :param num_hidden_list: list of integers indicating hidden nodes of each layer
        """
        super(LinearNetwork, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list

        # create list of linear layers
        # first hidden layer
        self.linear = [linear_torch.Linear(self.in_features, self.num_hidden_list[0])]
        # append additional hidden layers to list
        for idx in range(self.num_layers - 1):
            self.linear.append(
                linear_torch.Linear(self.num_hidden_list[idx], self.num_hidden_list[idx + 1]))

        # create ModuleList to make list of layers work
        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        """
        forward pass, which is same for conventional feed-forward net
        :param inputs: inputs with shape [batch_size, in_features]
        :return: logit outputs from the network
        """
        # first layer
        linear1 = F.relu(self.linear[0](inputs))

        linear2 = self.linear[1](linear1)

        return linear2
