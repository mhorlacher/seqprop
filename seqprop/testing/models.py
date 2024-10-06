import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- GpC Model 
def count_GpC(seq):
    """Count the occurences of the GpC motifs in a sequence. 

    Args:
        seq (torch.Tensor): 1D tensor of integers representing a sequence of nucleotides
                            with [A, C, G, T] -> [0, 1, 2, 3]. 

    Returns:
        torch.Tensor: Scalar count of GpC motifs in the sequence. 
    """
    count = 0
    for i in range(len(seq)):
        count += torch.equal(seq[i:(i+2)], torch.tensor([2, 1])) # GC -> [2, 1]
    return torch.tensor(count, dtype=torch.float32)

class GpCDataset(torch.utils.data.IterableDataset):
    def __init__(self, length=128, num_samples=1000, steepness=1.0):
        """Generate a dataset where the binary label is proportional to the number of GpC motifs in the sequence. 

        Args:
            length (int, optional): Sequence length. Defaults to 128.
            num_samples (int, optional): Number of samples in the dataset. Defaults to 1000.
            steepness (float, optional): How sensitive the label is to GpC content. Defaults to 1.0.
        """
        super().__init__()
        self.length = length
        self.num_samples = num_samples
        self.expected_gc = (length-1)*(1/4)**2
        self.steepness = steepness

    def generate_sample(self, return_target_probs=False):
        seq = torch.randint(0, 4, (self.length, ))
        target = F.sigmoid((count_GpC(seq) - self.expected_gc)*self.steepness) # center around 0
        if not return_target_probs:
            target = torch.bernoulli(target)
        return F.one_hot(seq).T, target

    def __iter__(self, return_target_probs=False):
        for _ in range(self.num_samples):
            x, y = self.generate_sample(return_target_probs)
            yield x.float(), y.float()

class GpCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv1d(48, 2) # in theory we only need one kernel
        self.pool = lambda x: torch.mean(x, dim=-1)
        self.fc1 = nn.LazyLinear(1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.fc1(x) # logits
        
        x = torch.squeeze(x, dim=-1)
        return x
    
    def load(self, ckpt_path: str | None = None):
        if ckpt_path is None:
            ckpt_path = importlib.resources.files('seqprop') / 'testing' / 'assets' / 'GpC_model.pt'
        self.load_state_dict(torch.load(ckpt_path))
        return self

    # def train(self, epochs=100, ds_length=128, ds_num_samples=1000, ds_steepness=1.0):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    #     criterion = nn.BCEWithLogitsLoss()

    #     for epoch in range(100):
    #         for i, (x, y) in enumerate(dl):
    #             optimizer.zero_grad()
    #             y_pred = model(x)
    #             loss = criterion(y_pred, y)
    #             loss.backward()
    #             optimizer.step()
    #             if i % 100 == 0:
    #                 print(f"epoch {epoch}, iter {i}, loss {loss.item()}")