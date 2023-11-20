import gc

from timeit import default_timer
from tqdm import tqdm
import wandb

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from data_utils.hdf5_datasets import Equation


def simple_trainer(
    model,
    train_loader,
    test_loader,
    params,
    wandb_log=False,
    log_test_interval=1,
    stage='ssl',
):

    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    epochs = params.epochs
    # weight_path = params.weight_path
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=False
    )
    scheduler = StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma)
    loss_p = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_iter = tqdm(
            train_loader,
            desc=f'Epoch {ep}/{epochs}',
            leave=False,
            ncols=100
        )
        for x, y in train_loader_iter:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            optimizer.zero_grad()
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            train_count += 1

            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            loss = loss_p(target.reshape(batch_size, -1),
                          out.reshape(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
            del x, y, out, loss
            gc.collect()

        torch.cuda.empty_cache()
        scheduler.step()
        t2 = default_timer()
        epoch_train_time = t2 - t1
        avg_train_l2 = train_l2 / train_count

        if ep % log_test_interval == 0:

            values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
            print(f"Epoch {ep}: "
                  f"Time: {epoch_train_time:.2f}s, "
                  f"Loss: {avg_train_l2:.4f}")

            if wandb_log:
                wandb.log(values_to_log, step=ep, commit=True)

    # torch.save(model.state_dict(), weight_path)

    model.eval()
    test_l2 = 0.0
    ntest = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            out, _, _, _ = model(x)
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
    batch_index: int,
):
    slices = [
        batch_index,
        slice(None),
        slice(None),
        slice(None),
    ]

    # flat "projection" to attend to only the fields relevant
    # to the equation under study.
    if eq == Equation.SWE:
        slices.append(0)
    elif eq == Equation.DIFF:
        slices.append(slice(1, 3))
    elif eq == Equation.NS:
        slices.append(slice(3, None))
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
    params,
    epochs=None,
    wandb_log=False,
    log_interval=1,
    stage='ssl',
):
    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    if epochs is None:
        epochs = params.epochs
    # weight_path = params.weight_path
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=False,
    )
    scheduler = StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma,
    )
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_tqdm = tqdm(
            train_loader,
            desc=f'Epoch {ep}/{epochs}',
            leave=False,
            ncols=100
        )
        for x, y in train_loader_tqdm:
            equations = x[1]
            x = x[0].cuda()
            y = y[0].cuda()
            optimizer.zero_grad()

            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            train_count += 1

            if params.pretrain_ssl:
                target = x.clone()
            else:
                target = y.clone()

            # Could this be made more efficient by collecting
            # the same equations in one "mini-batch?"
            for k, eq in enumerate(equations):
                loss = multi_physics_loss(
                    target, 
                    out, 
                    loss_fn, 
                    Equation(eq.item()),
                    batch_index=k,
                )
                loss.backward()
                train_l2 += loss.item()

            # Clip gradients to prevent exploding gradients:
            if params.gradient['clip']:
                nn.utils.clip_grad_value_(
                    model.parameters(),
                    params.gradient['threshold'],
                )

            optimizer.step()
            del x, y, out, loss
            gc.collect()

        torch.cuda.empty_cache()
        scheduler.step()

        if ep % log_interval == 0:
            t2 = default_timer()
            epoch_train_time = t2 - t1
            avg_train_l2 = train_l2 / train_count
            print(f"Epoch {ep}: | "
                  f"Time: {epoch_train_time:.2f}s | "
                  f"Loss: {avg_train_l2:.4f}")

            if wandb_log:
                values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
                wandb.log(values_to_log, step=ep, commit=True)

    # torch.save(model.state_dict(), weight_path)

    model.eval()
    t1 = default_timer()
    test_l2 = 0.0
    n_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            equations = y[1]
            x = x[0].cuda()
            y = y[0].cuda()

            out, _, _, _ = model(x)
            n_test += 1
            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            for k, eq in enumerate(equations):
                loss = multi_physics_loss(
                    target, 
                    out, 
                    loss_fn, 
                    Equation(eq.item()),
                    batch_index=k,
                )
                test_l2 += loss.item()

    test_l2 /= n_test
    t2 = default_timer()
    test_time = t2 - t1
    print(f"Time: {test_time:.2f}s\n"
          f"Loss: {test_l2:.6f}")

    if wandb_log:
        wandb.log({'test_error': test_l2}, commit=True)
