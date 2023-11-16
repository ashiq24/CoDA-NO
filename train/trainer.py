import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from data_utils.data_utils import *
from timeit import default_timer
import gc
from tqdm import tqdm
import wandb


def simple_trainer(
        model,
        train_loader,
        test_loader,
        params,
        wandb_log=False,
        log_test_interval=1,
        normalizer=None,
        stage='ssl'):

    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    epochs = params.epochs
    weight_path = params.weight_path
    optimizer = Adam(model.parameters(), lr=lr,
                     weight_decay=weight_decay, amsgrad=False)
    scheduler = StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma)
    loss_p = nn.MSELoss()
    loss_p1 = nn.L1Loss()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_iter = tqdm(
            train_loader,
            desc=f'Epoch {ep}/{epochs}',
            leave=False,
            ncols=100)
        for x, y in train_loader_iter:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]

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

            optimizer.zero_grad()

            out = model(x, out_grid_displacement, in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]
            train_count += 1

            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            loss_l2 = loss_p(target.reshape(batch_size, -1), out.reshape(batch_size, -1)
                             ) / torch.norm(target.reshape(batch_size, -1), p=2, dim=-1)
            loss_l1 = loss_p1(target.reshape(batch_size, -1), out.reshape(batch_size, -1)
                              ) / torch.norm(target.reshape(batch_size, -1), p=1, dim=-1)
            loss = 0.5 * loss_l1 + 0.5 * loss_l2
            loss.backward()

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_value_(model.parameters(), 0.001)

            optimizer.step()
            train_l2 += loss_l2.item()
            del x, y, out, loss
            gc.collect()

        torch.cuda.empty_cache()
        scheduler.step()
        t2 = default_timer()
        epoch_train_time = t2 - t1
        avg_train_l2 = train_l2 / train_count

        if ep % log_test_interval == 0:

            values_to_log = {
                'train_err_' + stage: avg_train_l2,
                'time_' + stage: epoch_train_time}
            print(
                f"Epoch {ep}: Time: {epoch_train_time:.2f}s, Loss {stage}: {avg_train_l2:.6f}")

            wandb.log(values_to_log, commit=True)

    torch.save(model.state_dict(), weight_path)

    model.eval()
    test_l2 = 0.0
    ntest = 0
    with torch.no_grad():
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

            batch_size = x.shape[0]
            out = model(x, out_grid_displacement, in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]

            ntest += 1
            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            test_l2 += loss_p(target.reshape(batch_size, -1), out.reshape(batch_size, -1)
                              ).item() / torch.norm(target.reshape(batch_size, -1), p=2, dim=-1).item()

    test_l2 /= ntest
    t2 = default_timer()

    wandb.log({'test_error_' + stage: test_l2}, commit=True)
    print(f"Test Error  {stage}: ", test_l2)
