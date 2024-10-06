import logging

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

class OneHotCategorigalSTE(torch.autograd.Function):
    """Implements a Straight-Through Estimator for the Categorical distribution.
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def forward(logits: torch.Tensor, n: int):
        """Samples from a one-hot categorical distribution according to the logits. 

        Args:
            logits (torch.Tensor): Logits of shape (batch_size, num_classes). 
            n (int): Number of samples to draw. 

        Returns:
            torch.Tensor: Samples of shape (n, batch_size, num_classes). 
        """
        output = OneHotCategorical(logits=logits).sample((n, )).float()

        logging.debug(f"{logits.shape=}")
        logging.debug(f"{output.shape=}")

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        """Passes the (averaged) gradients through. 

        Args:
            ctx (Any): Context object, unused. 
            grad_outputs (torch.Tensor): Gradients of the samples of shape (n, batch_size, num_classes). 

        Returns:
            torch.Tensor: Estimated gradients of the logits of shape (batch_size, num_classes). 
        """

        grad_outputs_sum = grad_outputs.detach().sum(dim=0) # Sum over the samples. 

        # NOTE: Why do we sum instead of averaging? The targets of the model are already 
        # averaged over the samples, therefore we need to sum in order to not downscale the 
        # gradients. 

        logging.debug(f"{grad_outputs.shape=}")
        logging.debug(f"{grad_outputs_sum.shape=}")

        return grad_outputs_sum, None # No gradients for 'n', the number of samples. 


class OneHotCategoricalSampler(torch.nn.Module):
    """Implements a differentiable categorical sampling layer. 
    """

    def __init__(self, n: int = 1) -> None:
        super().__init__()

        self.n = n
    
    def forward(self, logits):
        # TODO: Add param for number of samples.
        # TODO: Add param for temperature. 

        logging.debug(f"{self.n=}")

        return OneHotCategorigalSTE.apply(logits, self.n)