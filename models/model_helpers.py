import torch


def count_parameters(model):
    with torch.no_grad():
        total_count = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            pcount = torch.tensor(p.numel())
            total_count += int(pcount.item())

    return total_count / 1e6
