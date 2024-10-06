"""Implementation of differentiable penalties for sequence optimization. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from seqprop.templating import sequence_to_multihot

class DoubleGpCPenalty(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(4, 1, 4, bias=True)
        self.conv1d.weight = nn.Parameter(sequence_to_multihot('GCGC').float().unsqueeze(0).transpose(1, 2))
        self.conv1d.bias = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, inputs):
        return F.relu(self.conv1d(inputs.transpose(1, 2))).sum(dim=-1).mean()