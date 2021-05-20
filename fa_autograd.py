# This is an adoption from pytorch tutorial
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function


class RegLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, backprop_weight, regularization):
        ctx.save_for_backward(input, weight, backprop_weight, regularization)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, backprop_weight, regularization = ctx.saved_tensors
        grad_input = grad_weight = grad_backprop_weight = grad_regularization = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input) + regularization * weight

        return grad_input, grad_weight, grad_backprop_weight, grad_regularization


reg_linear = RegLinearFunction.apply


class RegLinear(nn.Module):
    def __init__(self, input_features, output_features, regularization=0):
        super(RegLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.empty(output_features, input_features))
        self.backprop_weight = nn.Parameter(Variable(torch.FloatTensor(
            output_features, input_features), requires_grad=False))
        self.regularization = nn.Parameter(Variable(
            regularization * torch.ones_like(self.weight), requires_grad=False))

        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.backprop_weight)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return reg_linear(input, self.weight, self.backprop_weight, self.regularization)


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
        grad_input = grad_weight = grad_backprop_weight = None
        def act_derivative(x): return torch.relu(torch.sign(x))
        if ctx.needs_input_grad[0]:
            grad_input = (act_derivative(input.mm(weight.t()))
                          * grad_output).mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (act_derivative(input.mm(weight.t())
                                          ).t() * grad_output.t()).mm(input)

        return grad_input, grad_weight, grad_backprop_weight


fa_function_relu = FeedbackAlignmentFunctionReLU.apply


class FeedbackAlignmentReLU(nn.Module):
    def __init__(self, input_features, output_features):
        super(FeedbackAlignmentReLU, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(output_features, input_features))
        self.backprop_weight = nn.Parameter(Variable(torch.FloatTensor(
            output_features, input_features), requires_grad=False))

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
        grad_input = grad_weight = grad_backprop_weight = None
        def act_derivative(x): return torch.sigmoid(x)(torch.ones_like(x) - torch.sigmoid(x))
        if ctx.needs_input_grad[0]:
            grad_input = (act_derivative(input.mm(weight.t()))
                          * grad_output).mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (act_derivative(input.mm(weight.t())
                                          ).t() * grad_output.t()).mm(input)

        return grad_input, grad_weight, grad_backprop_weight


fa_function_sigmoid = FeedbackAlignmentFunctionSigmoid.apply


class FeedbackAlignmentSigmoid(nn.Module):
    def __init__(self, input_features, output_features):
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
        grad_input = grad_weight = grad_backprop_weight = None
        def act_derivative(x): return torch.ones_like(x)
        if ctx.needs_input_grad[0]:
            grad_input = (act_derivative(input.mm(weight.t()))
                          * grad_output).mm(backprop_weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (act_derivative(input.mm(weight.t())
                                          ).t() * grad_output.t()).mm(input)

        return grad_input, grad_weight, grad_backprop_weight


fa_function_linear = FeedbackAlignmentFunctionLinear.apply


class FeedbackAlignmentLinear(nn.Module):
    def __init__(self, input_features, output_features):
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
