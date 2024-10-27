"""Implementation of differentiable penalties for sequence optimization. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from seqprop.templating import sequence_to_multihot

class DoubleGpCPenalty(nn.Module):
    def __init__(self, motif_multihot: torch.Tensor):
        super().__init__()
        self.conv1d = nn.Conv1d(4, 1, 4, bias=True)
        self.conv1d.weight = nn.Parameter(sequence_to_multihot('GCGC').float().unsqueeze(0).transpose(1, 2))
        self.conv1d.bias = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, inputs):
        return F.relu(self.conv1d(inputs.transpose(1, 2))).sum(dim=-1).mean()

class MotifPenalty(nn.Module):
    def __init__(self, motif_multihot: torch.Tensor, exclude_positions: list[int] | None = None):
        super().__init__()
        assert len(motif_multihot.shape) == 2
        self.exclude_positions = exclude_positions

        self.conv1d = nn.Conv1d(motif_multihot.shape[1], 1, motif_multihot.shape[0], bias=True)
        self.conv1d.weight = nn.Parameter(motif_multihot.float().unsqueeze(0).transpose(1, 2))
        self.conv1d.bias = nn.Parameter(torch.tensor([torch.tensor(-motif_multihot.shape[0] + 1, dtype=torch.float32)]))

    def forward(self, inputs):
        return F.relu(self.conv1d(inputs.transpose(1, 2))).squeeze(1).sum(dim=-1)
    
    @classmethod
    def from_sequence(cls, motif_sequence: str):
        return cls(sequence_to_multihot(motif_sequence))