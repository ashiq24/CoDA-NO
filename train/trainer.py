import gc

from timeit import default_timer
import tqdm
import wandb
from data_utils.data_utils import *
import torch
from torch import nn
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from data_utils.hdf5_datasets import Equation


MAP_EQUATION_TO_CHANNELS = {
    Equation.SWE: (0,),
    Equation.DIFF: (1, 2),
    Equation.NS: (3, 4, 5,),
}


def simple_trainer(
    model,
    train_loader,
    test_loader,
    params,
    wandb_log=False,
    log_test_interval=1,
    stage='ssl',
    normalizer=None,
):

    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    epochs = params.epochs
    weight_path = params.weight_path
    optimizer = Adam(model.parameters(), lr=lr,
                     weight_decay=weight_decay, amsgrad=False)
    if params.scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_step,
            gamma=scheduler_gamma)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=scheduler_step,
            factor=scheduler_gamma)

    loss_p = nn.MSELoss()
    loss_p1 = nn.L1Loss()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_iter = tqdm.tqdm(
            train_loader,
            desc=f'Epoch {ep}/{epochs}',
            leave=False,
            ncols=100
        )
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

            out = model(
                x,
                out_grid_displacement=out_grid_displacement,
                in_grid_displacement=in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]
            
            #print('Shapes', out.shape, x.shape)
            train_count += 1

            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            loss = loss_p(target.reshape(batch_size, -1),
                          out.reshape(batch_size, -1))
            loss.backward()

            # Clip gradients to prevent exploding gradients
            if params.clip_gradient:
                nn.utils.clip_grad_value_(
                    model.parameters(), params.gradient_clip_value)

            optimizer.step()
            train_l2 += loss.item()
            del x, y, out, loss
            gc.collect()

        torch.cuda.empty_cache()
        avg_train_l2 = train_l2 / train_count
        if params.scheduler_type != 'step':
            scheduler.step(avg_train_l2)
        else:
            scheduler.step(avg_train_l2)
        t2 = default_timer()
        epoch_train_time = t2 - t1

        if ep % log_test_interval == 0:

            values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
            print(f"Epoch {ep}: "
                  f"Time: {epoch_train_time:.2f}s, "
                  f"Loss: {avg_train_l2:.4f}")

            wandb.log(values_to_log, commit=True)
    weight_path = weight_path + params.config + "_" + stage+'.pt'
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
            out, _, _, _ = model(x, in_grid_displacement=in_grid_displacement,out_grid_displacement=out_grid_displacement)
            
            ntest += 1
            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            test_l2 += loss_p(target.reshape(batch_size, -1),
                              out.reshape(batch_size, -1)).item()

    test_l2 /= ntest
    t2 = default_timer()

    if wandb_log:
        wandb.log({'test_error': test_l2}, commit=True)
    print("Test Error : ", test_l2)


def multi_physics_loss(
    ground_truth,
    prediction,
    loss_fn,
    eq: Equation,
):
    slices = [
        slice(None),
        slice(None),
        slice(None),
    ]

    # flat "projection" to attend to only the fields relevant
    # to the equation under study.
    if eq == Equation.SWE:
        slices.insert(0, 0)
    elif eq == Equation.DIFF:
        slices.insert(0, slice(1, 3))
    elif eq == Equation.NS:
        slices.insert(0, slice(3, None))
    else:
        raise ValueError(f"Invalid equation: {eq}")

    return loss_fn(
        ground_truth[slices].reshape(1, -1),
        prediction[slices].reshape(1, -1),
    )


# Could this be refactored with `simple_trainer` above?
def multi_physics_trainer(
    model,
    train_loader,
    test_loader,
    loss_fn,
    params,
    epochs=None,
    wandb_log=False,
    log_interval=1,
    script=True,
):
    if epochs is None:
        epochs = params.epochs
    # weight_path = params.weight_path
    optimizer = Adam(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
        amsgrad=False,
    )
    scheduler = StepLR(
        optimizer,
        step_size=params.scheduler_step,
        gamma=params.scheduler_gamma,
    )
    # loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        _tqdm = tqdm.tqdm if script else tqdm.tqdm_notebook
        train_loader_tqdm = _tqdm(
            train_loader,
            desc=f'Epoch {ep}/{epochs}',
            leave=False,
            # ncols=100,
        )
        for x, y in train_loader_tqdm:
            equations = x[1]
            x = x[0].cuda()
            y = y[0].cuda()
            optimizer.zero_grad()

            model.next_channels = tuple([
                MAP_EQUATION_TO_CHANNELS[Equation(eq.item())] for eq in equations
            ])
            out, *_ = model(x, equations=equations)
            train_count += 1  # i think this should be `+= batch_size`

            losses = torch.tensor(0.0, dtype=torch.float).cuda()
            # Could this be made more efficient by collecting
            # the same equations in one "mini-batch?"
            for k, eq in enumerate(equations):
                # loss = multi_physics_loss(
                #     y[k],
                #     out[k],
                #     loss_fn, 
                #     Equation(eq.item()),
                # )
                loss = loss_fn(
                    y[k].view(1, -1),
                    out[k].view(1, -1),
                )
                losses += loss

            losses.backward()
            train_l2 += losses.item()

            # Clip gradients to prevent exploding gradients:
            if params.gradient['clip']:
                nn.utils.clip_grad_value_(
                    model.parameters(),
                    params.gradient['threshold'],
                )

            optimizer.step()
            del x, y, out, losses
            gc.collect()

        torch.cuda.empty_cache()
        scheduler.step()

        if ep % log_interval == 0:
            t2 = default_timer()
            epoch_train_time = t2 - t1
            avg_train_l2 = train_l2 / train_count
            # print(f"{train_l2=}", f"{train_count=}", f"{avg_train_l2=}", sep='\n')
            print(f"Epoch {ep}: | "
                  f"Time: {epoch_train_time:.2f}s | "
                  f"Loss: {avg_train_l2:.4f}")

            if wandb_log:
                # TODO help wb.log() handle multi-stage trainings
                # With the current reconstructive/predictive training phases,
                # W&B rejects logs from all epochs that are "out of order"
                # (i.e. all phases after the first).
                values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
                wandb.log(values_to_log, step=ep, commit=True)

    # torch.save(model.state_dict(), weight_path)

    model.eval()
    t1 = default_timer()
    test_l2 = 0.0
    # Counts how many data points we've tested against.
    # There may be multiple per batch.
    n_test = 0
    _tqdm = tqdm.tqdm if script else tqdm.tqdm_notebook
    test_loader_tqdm = _tqdm(
        test_loader,
        desc=f'Final Exam ({epochs} / {epochs})',
        leave=False,
        ncols=100
    )
    with torch.no_grad():
        for x, y in test_loader_tqdm:
            equations = y[1]
            x = x[0].cuda()
            y = y[0].cuda()

            model.next_channels = tuple([
                MAP_EQUATION_TO_CHANNELS[Equation(eq.item())] for eq in equations
            ])
            out, *_ = model(x)

            for k, eq in enumerate(equations):
                loss = multi_physics_loss(
                    y[k],
                    out[k],
                    loss_fn, 
                    Equation(eq.item()),
                )
                test_l2 += loss.item()
                n_test += 1

    test_l2 /= n_test
    t2 = default_timer()
    test_time = t2 - t1
    print(f"Time: {test_time:.2f}s\n"
          f"Loss: {test_l2:.6f}")

    if wandb_log:
        wandb.log({'test_error': test_l2}, commit=True)


# doesn't "know" what a single physics is,
# but it at least knows how to start and stop at given indices.
def test_single_physics(
    model: nn.Module,
    test_loader: data.DataLoader,
    loss_fn,
    start: int,
    stop: int,
    script=True
) -> None:
    # loss_fn = nn.MSELoss()
    t1 = default_timer()
    test_l2 = 0.0
    n_test = 0

    with torch.no_grad():
        # TODO TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`
        _trange = tqdm.trange if script else tqdm.tnrange
        test_loader_trange = _trange(
            start,
            stop,
            desc=f'Testing [{start} : {stop}]',
            leave=False,
            ncols=120,
        )

        for i in test_loader_trange:
            x, y = test_loader.dataset[i]
            eq = x[1]
            x = x[0].unsqueeze(0).cuda()
            y = y[0].unsqueeze(0).cuda()

            model.next_channels = (MAP_EQUATION_TO_CHANNELS[Equation(eq)],)
            out, *_ = model(x)
            loss = multi_physics_loss(
                y[0],
                out[0],
                loss_fn,
                Equation(eq),
            )
            test_l2 += loss.item()
            n_test += 1

    t2 = default_timer()
    test_time = t2 - t1
    print(f"Time: {test_time:.2f}s\n"
          f"Loss: {test_l2 / n_test:.6f}")

