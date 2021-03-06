import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import fa_autograd


class TwoLayerFeedbackAlignmentNetworkReLU(nn.Module):
    def __init__(self, input_features, hidden_features, regularization):
        super(TwoLayerFeedbackAlignmentNetworkReLU, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentReLU(
            self.input_features, self.hidden_features)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, regularization)

    def forward(self, X):
        hidden = self.first_layer(X)
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentNetworkSigmoid(nn.Module):
    def __init__(self, input_features, hidden_features, regularization):
        super(TwoLayerFeedbackAlignmentNetworkSigmoid, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentSigmoid(
            self.input_features, self.hidden_features)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, regularization)

    def forward(self, X):
        hidden = self.first_layer(X)
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentNetworkTanh(nn.Module):
    def __init__(self, input_features, hidden_features, regularization):
        super(TwoLayerFeedbackAlignmentNetworkTanh, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentTanh(
            self.input_features, self.hidden_features)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, regularization)

    def forward(self, X):
        hidden = self.first_layer(X)
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerBackPropNetworkReLU(nn.Module):
    def __init__(self, input_features, hidden_features):
        super(TwoLayerBackPropNetworkReLU, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = nn.Linear(
            input_features, hidden_features, bias=False)
        self.second_layer = nn.Linear(hidden_features, 1, bias=False)
        nn.init.normal_(self.first_layer.weight)
        nn.init.normal_(self.second_layer.weight)

    def forward(self, X):
        hidden = F.relu(self.first_layer(X))
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentDropoutNetworkReLU(nn.Module):
    def __init__(self, input_features, hidden_features, dropout_prob):
        super(TwoLayerFeedbackAlignmentDropoutNetworkReLU, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentReLU(
            self.input_features, self.hidden_features)
        self.drop = nn.Dropout(dropout_prob)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, 0)

    def forward(self, X):
        hidden = self.first_layer(X)
        hidden_dropped = self.drop(hidden)
        prediction = self.second_layer(
            hidden_dropped) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentDropoutNetworkSigmoid(nn.Module):
    def __init__(self, input_features, hidden_features, dropout_prob):
        super(TwoLayerFeedbackAlignmentDropoutNetworkSigmoid, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentSigmoid(
            self.input_features, self.hidden_features)
        self.drop = nn.Dropout(dropout_prob)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, 0)

    def forward(self, X):
        hidden = self.first_layer(X)
        hidden_dropped = self.drop(hidden)
        prediction = self.second_layer(
            hidden_dropped) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentDropoutNetworkTanh(nn.Module):
    def __init__(self, input_features, hidden_features, dropout_prob):
        super(TwoLayerFeedbackAlignmentDropoutNetworkTanh, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.FeedbackAlignmentTanh(
            self.input_features, self.hidden_features)
        self.drop = nn.Dropout(dropout_prob)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, 0)

    def forward(self, X):
        hidden = self.first_layer(X)
        hidden_dropped = self.drop(hidden)
        prediction = self.second_layer(
            hidden_dropped) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentNetworkLinear(nn.Module):
    def __init__(self, input_features, hidden_features, regularization):
        super(TwoLayerFeedbackAlignmentNetworkLinear, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.RegLinear(
            self.input_features, self.hidden_features)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, regularization)

    def forward(self, X):
        hidden = self.first_layer(X)
        prediction = self.second_layer(hidden) / np.sqrt(self.hidden_features)
        return prediction


class TwoLayerFeedbackAlignmentDropoutNetworkLinear(nn.Module):
    def __init__(self, input_features, hidden_features, dropout_prob):
        super(TwoLayerFeedbackAlignmentDropoutNetworkLinear, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features

        self.first_layer = fa_autograd.RegLinear(
            self.input_features, self.hidden_features)
        self.drop = nn.Dropout(dropout_prob)
        self.second_layer = fa_autograd.RegLinear(
            self.hidden_features, 1, 0)

    def forward(self, X):
        hidden = self.first_layer(X)
        hidden_dropped = self.drop(hidden)
        prediction = self.second_layer(
            hidden_dropped) / np.sqrt(self.hidden_features)
        return prediction
