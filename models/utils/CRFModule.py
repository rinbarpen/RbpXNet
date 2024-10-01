import torch
from torch import nn
from torch import optim

class CRF(nn.Module):
    def __init__(self, tag_to_ix, n_features):
        super(CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.n_features = n_features
        self.transition = nn.Parameter(torch.Tensor(self.tag_to_ix.keys(), self.tag_to_ix.keys()))
    def forward(self, input, mask):
        emission = self._get_emission(input)
        return self._crf_loss(emission, mask, self.transition)
