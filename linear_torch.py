# This is an adoption from pytorch tutorial
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function


class FeedbackAlignmentFunctionReLU(Function):
    @staticmethod
    def forward(ctx, input, weight, backprop_weight):
        activation = torch.relu
        ctx.save_for_backward(input, weight, backprop_weight)
        output = activation(input.mm(weight.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, backprop_weight = ctx.saved_variables
        grad_input = grad_weight = None
        def act_derivative(x): return torch.relu(torch.sign(x))
        if ctx.needs_input_grad[0]:
            grad_input = (act_derivative(input.mm(weight.t()))
                          * grad_output).mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (act_derivative(input.mm(weight.t())
                                          ).t() * grad_output.t()).mm(input)

        return grad_input, grad_weight


fa_function_relu = FeedbackAlignmentFunctionReLU.apply


class FeedbackAlignmentReLU(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(FeedbackAlignmentReLU, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(output_features, input_features))
        self.backprop_weight = Variable(torch.FloatTensor(
            output_features, input_features), requires_grad=False)

        # weight initialization
        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.backprop_weight)

    def forward(self, input):
        return fa_function_relu(input, self.weight, self.backprop_weight)


class FeedbackAlignmentFunctionSigmoid(Function):
    @staticmethod
    def forward(ctx, input, weight, backprop_weight):
        activation = torch.sigmoid
        ctx.save_for_backward(input, weight, backprop_weight)
        output = activation(input.mm(weight.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, backprop_weight = ctx.saved_variables
        grad_input = grad_weight = None
        def act_derivative(x): return torch.sigmoid(x)(1 - torch.sigmoid(x))
        if ctx.needs_input_grad[0]:
            grad_input = (act_derivative(input.mm(weight.t()))
                          * grad_output).mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (act_derivative(input.mm(weight.t())
                                          ).t() * grad_output.t()).mm(input)

        return grad_input, grad_weight


fa_function_sigmoid = FeedbackAlignmentFunctionSigmoid.apply


class FeedbackAlignmentSigmoid(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(FeedbackAlignmentSigmoid, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(output_features, input_features))
        self.backprop_weight = Variable(torch.FloatTensor(
            output_features, input_features), requires_grad=False)

        # weight initialization
        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.backprop_weight)

    def forward(self, input):
        return fa_function_sigmoid(input, self.weight, self.backprop_weight)


class FeedbackAlignmentFunctionLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, backprop_weight):
        def activation(x): return x
        ctx.save_for_backward(input, weight, backprop_weight)
        output = activation(input.mm(weight.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, backprop_weight = ctx.saved_variables
        grad_input = grad_weight = None
        def act_derivative(x): return torch.ones_like(x)
        if ctx.needs_input_grad[0]:
            grad_input = (act_derivative(input.mm(weight.t()))
                          * grad_output).mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (act_derivative(input.mm(weight.t())
                                          ).t() * grad_output.t()).mm(input)

        return grad_input, grad_weight


fa_function_linear = FeedbackAlignmentFunctionLinear.apply


class FeedbackAlignmentLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(FeedbackAlignmentLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(output_features, input_features))
        self.backprop_weight = Variable(torch.FloatTensor(
            output_features, input_features), requires_grad=False)

        # weight initialization
        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.backprop_weight)

    def forward(self, input):
        return fa_function_linear(input, self.weight, self.backprop_weight)
