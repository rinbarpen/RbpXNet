import math
import torch
from torch import nn


class WeightGate(nn.Module):
    def __init__(self):
        super(WeightGate, self).__init__()

        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a, b are (B, C, X..)
        weight = torch.sigmoid(self.weight)
        weight = weight.to(a.device)

        return a * weight, b * (1.0 - weight)


class DirectMultiWeightGate(nn.Module):
    def __init__(self, num_tensor: int):
        super(DirectMultiWeightGate, self).__init__()

        self.weights = nn.Parameter(torch.randn(num_tensor))

    def forward(self, *x: torch.Tensor):
        # x are (B, C, X..)
        weights = torch.sigmoid(self.weights)
        weights = weights.to(x[0].device)

        weighted_outputs = []
        for i, tensor in enumerate(x):
            weighted_outputs.append(tensor * weights[i])
        return weighted_outputs


class TinyMultiWeightGate(nn.Module):
    def __init__(self, num_tensor: int):
        super(TinyMultiWeightGate, self).__init__()
        assert num_tensor > 0

        self.weights = nn.Parameter(torch.randn(int(math.log(num_tensor)) + 1))

    def forward(self, *x: torch.Tensor):
        # x are (B, C, X..)
        assert len(x) >= 2

        weights = torch.sigmoid(self.weights)
        weights = weights.to(x[0].device)

        offset = len(x) // len(weights)

        begin = 0
        end = offset
        weighted_outputs = []
        for weight in weights:
            weighted_outputs.extend(x[begin:end] * weight)
            begin = end
            end = len(x) if end + offset > len(x) else end + offset

        return weighted_outputs
