import torch

# From onmt-py
def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(
        lengths.unsqueeze(1)))
