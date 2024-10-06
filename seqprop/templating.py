"""Utilities for creating SeqProp sequence templates. 
"""

import torch
import torch.nn.functional as F

from seqprop.plot import pwm_logo

extended_nt_map = {
    'A': {'A'},
    'C': {'C'},
    'G': {'G'},
    'U': {'U', 'T'},
    'T': {'U', 'T'},
    'R': {'A', 'G'},
    'Y': {'C', 'U', 'T'},
    'S': {'G', 'C'},
    'W': {'A', 'U', 'T'},
    'K': {'G', 'U', 'T'},
    'M': {'A', 'C'},
    'B': {'C', 'G', 'U', 'T'},
    'D': {'A', 'G', 'U', 'T'},
    'H': {'A', 'C', 'U', 'T'},
    'V': {'A', 'C', 'G'},
    'N': {'A', 'C', 'G', 'U', 'T'},
    '.': {'A', 'C', 'G', 'U', 'T'},
}
base2onehot = {
    'A': [1, 0, 0, 0], 
    'C': [0, 1, 0, 0], 
    'G': [0, 0, 1, 0], 
    'U': [0, 0, 0, 1], 
    'T': [0, 0, 0, 1],
}

def _nts_to_multihot(nucleotides: str) -> torch.Tensor:
    """Converts a nucleotide sequence to a one-hot tensor. 

    Args:
        nucleotides (str): The nucleotide sequence to convert. 

    Returns:
        torch.Tensor: The multi-hot tensor. 
    """
    return torch.tensor([base2onehot[nt] for nt in nucleotides]).max(dim=0).values

def sequence_to_multihot(sequence: str) -> torch.Tensor:
    """Converts a sequence to a multi-hot tensor. 

    Args:
        sequence (str): The sequence to convert. 

    Returns:
        torch.Tensor: The multi-hot tensor. 
    """
    return torch.stack([_nts_to_multihot(extended_nt_map[c]) for c in sequence])

class Template(torch.Tensor):
    def __new__(cls, data):
        return super().__new__(cls, data)

    @classmethod
    def from_sequence(cls, sequence: str) -> None:
        """Generates a template from a extended nucleotide sequence alphabet sequence. 

        Args:
            sequence (str): The sequence to generate the template from. 
        """
        return cls.__new__(cls, sequence_to_multihot(sequence))
    
    def show(self, **kwargs):
        return pwm_logo(self.numpy() / self.sum(dim=1).unsqueeze(-1).numpy(), **kwargs)
