import torch


def count_parameters(model):
    with torch.no_grad():
        total_count = 0
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: The total number of trainable parameters in millions.
    """
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pcount = torch.tensor(p.numel())
        total_count += int(pcount.item())

    return total_count / 1e6
