import torch
from data_utils.data_utils import *
import torch.nn as nn
from timeit import default_timer
import gc
from tqdm import tqdm
import wandb


def missing_variable_testing(
        model,
        test_loader,
        augmenter,
        normalizer,
        stage,
        params):
    print('Evaluating for Stage: ', stage)
    with torch.no_grad():
        ntest = 0
        test_l2 = 0
        loss_p = nn.MSELoss()
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            if params.grid_type == "non uniform":
                '''
                Assume non uniform grids requires
                updating grid for every sample. We need to
                suppy the grid.

                last 3 channel is displacement, taking (x,y), z is 0
                '''
                with torch.no_grad():
                    if stage == 'ssl':
                        out_grid_displacement = get_mesh_displacement(x)
                        in_grid_displacement = get_mesh_displacement(x)
                    else:
                        out_grid_displacement = get_mesh_displacement(y)
                        in_grid_displacement = get_mesh_displacement(x)
            else:
                out_grid_displacement = None
                in_grid_displacement = None

            if normalizer is not None:
                with torch.no_grad():
                    x, y = normalizer(x), normalizer(y)

            if augmenter is not None:
                x, _ = batched_masker(x, augmenter)

            batch_size = x.shape[0]
            out = model(x, out_grid_displacement, in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]
            ntest += 1

            target = y.clone()

            test_l2 += loss_p(target.reshape(batch_size, -1), out.reshape(batch_size, -1)
                              ).item() / torch.norm(target.reshape(batch_size, -1), p=2, dim=-1).item()

    test_l2 /= ntest
    t2 = default_timer()

    wandb.log({'Augmented test_error_' + stage: test_l2}, commit=True)
    print(f"Augmented Test Error  {stage}: ", test_l2)
