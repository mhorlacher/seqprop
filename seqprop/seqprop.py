import logging

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
    
from seqprop.ste import OneHotCategoricalSampler

class FastSeqProp(nn.Module):
    def __init__(self, template: torch.Tensor, n_samples=1, norm: nn.Module | None = None, device='cpu'):
        """FastSeqProp is a differentiable sequence design algorithm that uses a template to constrain the design space.

        Args:
            template (torch.Tensor): A template tensor of shape (seq_length, num_tokens) that constrains the design space.
            n_samples (int, optional): Number of samples generated in each batch during the optimization step. Defaults to 1.
            norm (nn.Module | None, optional): Normalization layer. Defaults to nn.LayerNorm.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.

        Raises:
            ValueError: If the template contains positions without a single valid token/nucleotide. 
        """
        super().__init__()
        self.template = template.detach().to(device)
        
        # check if there are any positions that do not contain a single valid token/nucleotide
        if template.sum(dim=1).min() == 0:
            raise ValueError("The template contains positions without a single valid token/nucleotide.")

        self.device = device
        self.sampler = OneHotCategoricalSampler(n=n_samples)

        # layer to normalize the logits before sampling
        self.norm = norm
        if self.norm is None:
            self.norm = nn.LayerNorm(self.template.shape[-1])
        self.norm.to(self.device)
    
    def _score_fn(self, inputs: torch.tensor):
        """Score function that evaluates the generated sequences (to be implemented by the user). 

        Args:
            inputs (torch.tensor): Generated sequences of shape (n_samples, seq_length, num_tokens).
        """
        raise NotImplementedError
    
    def _mask_logits_by_template(self, logits: torch.tensor):
        """Masks out positions in the logits that are not allowed by the template.

        To perform mask in a differentiable way, we add a large negative number to the logits. 

        Args:
            logits (torch.tensor): Logits of shape (seq_length, num_tokens).

        Returns:
            torch.Tensor: Masked logits.
        """
        return logits - (1 - self.template) * torch.tensor(1e10)
    
    def forward(self, logits: torch.tensor):
        """Main optimization step. 

        Logits are used to seed the sampler, which generates sequences that are scored by the user-defined 
        scoring function (e.g. a neural network). 

        Args:
            logits (torch.tensor): Logits of shape (seq_length, num_tokens).

        Returns:
            torch.Tensor: Scalar score, averaged over the samples. 
        """
        logits = self.norm(logits)

        # we mask out positions by adding a large negative number to the (normalized) logits
        logits = self._mask_logits_by_template(logits)

        return self._score_fn(self.sampler(logits)).mean()
    
    def design(self, steps: int = 100, maximize: bool = True, lr: float = 0.1, return_score_trace: bool = True):
        """Designs sequences that maximize (or minimizes) the score function.

        Args:
            steps (int, optional): Number of optimization steps. Defaults to 100.
            maximize (bool, optional): Whether to maximize or minimize the score. Defaults to True.
            lr (float, optional): Learning rate. Defaults to 0.1.
            return_score_trace (bool, optional): Whether to the score trace (for plotting). Defaults to True.

        Returns:
            tuple: Tuple containing the designed sequences, the final score, and the score trace (if requested). 
        """
        # randomly initialize the logits (we might want to play around with different initializations)
        logits = torch.rand(*self.template.shape, requires_grad=True, device=self.device, dtype=torch.float32)

        if return_score_trace:
            score_trace = []
        
        # initialize the optimizer
        parameters = [logits]
        if self.norm is not None:
            parameters += self.norm.parameters()
        optimizer = torch.optim.Adam(parameters, lr=lr, maximize=maximize)
        
        # main optimization loop
        for _ in tqdm(range(steps)):
            optimizer.zero_grad()
            score = self.forward(logits)
            score.backward()
            optimizer.step()
            
            if return_score_trace:
                score_trace.append(float(score.detach().cpu().numpy()))
        
        return (
            F.softmax(self._mask_logits_by_template(logits).detach().cpu(), dim=1).numpy(), 
            self.forward(logits).detach().cpu().numpy()
        ) + (score_trace if return_score_trace else None, )