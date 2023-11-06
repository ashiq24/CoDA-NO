import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
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
