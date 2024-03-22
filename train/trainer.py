import gc
import logging

from timeit import default_timer
import tqdm
import wandb
from data_utils.data_utils import *
import torch
from models.get_models import *
from torch import nn
from torch.utils import data
from .new_adam import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def nonuniform_mesh_trainer(
    model,
    train_loader,
    test_loader,
    params,
    wandb_log=False,
    log_test_interval=1,
    stage=StageEnum.RECONSTRUCTIVE,
    normalizer=None,
    variable_encoder=None,
    token_expander=None,
    initial_mesh=None,
):
    """
    Trains a model using a non-uniform mesh.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        test_loader (DataLoader): The data loader for the testing dataset.
        params (Namespace): The parameters for training.
        wandb_log (bool, optional): Whether to log training progress to wandb. Defaults to False.
        log_test_interval (int, optional): The interval at which to log test results. Defaults to 1.
        stage (StageEnum, optional): The stage of training. Defaults to StageEnum.RECONSTRUCTIVE.
        normalizer (object, optional): The normalizer object. Defaults to None.
        variable_encoder (object, optional): The variable encoder object. Defaults to None.
        token_expander (object, optional): The token expander object. Defaults to None.
        initial_mesh (object, optional): The initial mesh object. Defaults to None.

    Returns:
        float: The test error.
    """

    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    epochs = params.epochs
    weight_path = params.weight_path
    optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
    )
    if params.scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, patience=scheduler_step, factor=scheduler_gamma
        )

    loss_p = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_iter = tqdm.tqdm(
            train_loader, desc=f"Epoch {ep}/{epochs}", leave=False, ncols=100
        )
        for data in train_loader_iter:

            optimizer.zero_grad()

            x, y = data["x"], data["y"]

            static_features = data["static_features"]
            equation = [i[0] for i in data["equation"]]
            x, y = x.cuda(), y.cuda()
            if stage == StageEnum.RECONSTRUCTIVE:

                if params.masking:
                    x = model.do_mask(x)

            if variable_encoder is not None and token_expander is not None:
                inp = token_expander(
                    x,
                    variable_encoder(
                        initial_mesh + data["d_grid_x"].cuda()[0], equation
                    ),
                    static_features.cuda(),
                )

            elif params.n_static_channels > 0:
                inp = torch.cat(
                    [x, static_features[:, :, : params.n_static_channels].cuda()],
                    dim=-1,
                )
            else:
                inp = x

            batch_size = x.shape[0]

            if params.grid_type == "non uniform":
                """
                Assume non uniform grids requires
                updating grid for every sample. We need to
                suppy the grid.

                last 3 channel is displacement, taking (x,y), z is 0
                """
                with torch.no_grad():
                    if stage == StageEnum.RECONSTRUCTIVE:
                        out_grid_displacement = data["d_grid_x"].cuda()[0]
                        in_grid_displacement = data["d_grid_x"].cuda()[0]
                    else:
                        out_grid_displacement = data["d_grid_y"].cuda()[0]
                        in_grid_displacement = data["d_grid_x"].cuda()[0]
            else:
                out_grid_displacement = None
                in_grid_displacement = None

            out = model(
                inp,
                out_grid_displacement=out_grid_displacement,
                in_grid_displacement=in_grid_displacement,
            )

            if isinstance(out, (list, tuple)):
                out = out[0]

            train_count += 1

            if stage == StageEnum.RECONSTRUCTIVE:
                target = x.clone()

            else:
                target = y.clone()

            loss = loss_p(target.reshape(batch_size, -1), out.reshape(batch_size, -1))
            loss.backward()

            # Clip gradients to prevent exploding gradients
            if params.clip_gradient:
                nn.utils.clip_grad_value_(
                    model.parameters(), params.gradient_clip_value
                )

            optimizer.step()
            train_l2 += loss.item()
            del x, y, out, loss
            gc.collect()

        torch.cuda.empty_cache()
        avg_train_l2 = train_l2 / train_count
        if params.scheduler_type != "step":
            scheduler.step(avg_train_l2)
        else:
            scheduler.step(avg_train_l2)
        t2 = default_timer()
        epoch_train_time = t2 - t1

        if ep % log_test_interval == 0:

            values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
            print(
                f"Epoch {ep}: "
                f"Time: {epoch_train_time:.2f}s, "
                f"Loss: {avg_train_l2:.4f}"
            )

            if params.wandb_log:
                wandb.log(values_to_log, commit=True)

        # saving weights
        if ep % params.weight_saving_interval == 0 or ep == epochs - 1:
            stage_string = "ssl" if stage == StageEnum.RECONSTRUCTIVE else "sl"

            weight_path_model = (
                weight_path + params.config + "_" + stage_string + "_" + str(ep) + ".pt"
            )
            torch.save(model.state_dict(), weight_path_model)
            if variable_encoder is not None:
                variable_path = (
                    weight_path + params.config + "_variable_encoder_" + str(ep)
                )
                variable_encoder.save_all_encoder(variable_path)

    model.eval()
    test_l2 = 0.0
    ntest = 0
    with torch.no_grad():
        for data in test_loader:

            x, y = data["x"], data["y"]
            static_features = data["static_features"]
            equation = [i[0] for i in data["equation"]]
            x, y = x.cuda(), y.cuda()
            if variable_encoder is not None and token_expander is not None:
                inp = token_expander(
                    x,
                    variable_encoder(
                        initial_mesh + data["d_grid_x"].cuda()[0], equation
                    ),
                    static_features.cuda(),
                )
            elif params.n_static_channels > 0:
                inp = torch.cat(
                    [x, static_features[:, :, : params.n_static_channels].cuda()],
                    dim=-1,
                )
            else:
                inp = x

            if params.grid_type == "non uniform":
                """
                Assume non uniform grids requires
                updating grid for every sample. We need to
                suppy the grid.

                last 3 channel is displacement, taking (x,y), z is 0
                """
                with torch.no_grad():
                    if stage == StageEnum.RECONSTRUCTIVE:
                        out_grid_displacement = data["d_grid_x"].cuda()[0]
                        in_grid_displacement = data["d_grid_x"].cuda()[0]
                    else:
                        out_grid_displacement = data["d_grid_y"].cuda()[0]
                        in_grid_displacement = data["d_grid_x"].cuda()[0]
            else:
                out_grid_displacement = None
                in_grid_displacement = None

            batch_size = x.shape[0]
            out = model(
                inp,
                in_grid_displacement=in_grid_displacement,
                out_grid_displacement=out_grid_displacement,
            )
            if isinstance(out, (list, tuple)):
                out = out[0]

            ntest += 1
            if stage == StageEnum.RECONSTRUCTIVE:
                target = x.clone()
            else:
                target = y.clone()

            test_l2 += loss_p(
                target.reshape(batch_size, -1), out.reshape(batch_size, -1)
            ).item()

    test_l2 /= ntest
    t2 = default_timer()

    if params.wandb_log:
        wandb.log({"test_error_" + stage_string: test_l2}, commit=True)
    print("Test Error : " + stage_string, test_l2)
    return test_l2


############
# Followings are Training Routings for PDE bench dataset
############


def multi_physics_trainer(
    model,
    train_loader,
    test_loader,
    loss_fn,
    params,
    epochs=None,
    wandb_log=False,
    gradient_threshold_step_interval=10,
    logger=None,
    log_interval=1,
    script=True,
):
    """
    Trains a multi-physics model using the provided parameters.

    Args:
        model (nn.Module): The multi-physics model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        test_loader (DataLoader): The data loader for the test dataset.
        loss_fn (callable): The loss function used for training.
        params (Namespace): The parameters for training the model.
        epochs (int, optional): The number of training epochs. If not provided, it will be taken from params. Defaults to None.
        wandb_log (bool, optional): Whether to log training progress to Weights & Biases. Defaults to False.
        gradient_threshold_step_interval (int, optional): The interval at which to decrease the gradient threshold. Defaults to 10.
        logger (Logger, optional): The logger object for logging. Defaults to None.
        log_interval (int, optional): The interval at which to log training progress. Defaults to 1.
        script (bool, optional): Whether the training is running in a script or notebook. Defaults to True.

    Returns:
        None
    """
    if epochs is None:
        epochs = params.epochs
    batch_size = train_loader.batch_size
    gradient_threshold = params.gradient["threshold"]
    if logger is None:
        logger = logging.getLogger()

    optimizer = Adam(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
        amsgrad=False,
    )
    scheduler = StepLR(
        optimizer, step_size=params.scheduler_step, gamma=params.scheduler_gamma,
    )

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        _tqdm = tqdm.tqdm if script else tqdm.tqdm_notebook
        train_loader_tqdm = _tqdm(
            train_loader, desc=f"Epoch {ep}/{epochs}", leave=False,
        )
        for i, (x, y) in enumerate(train_loader_tqdm):
            equations = x[1]
            x = x[0].cuda()
            y = y[0].cuda()
            optimizer.zero_grad()

            out, *_ = model(x, equations=equations)
            train_count += batch_size

            losses = torch.tensor(0.0, dtype=torch.float).cuda()
            for k, eq in enumerate(equations):
                loss = loss_fn(y[k].view(1, -1), out[k].view(1, -1),)
                losses += loss

            losses.backward()
            train_l2 += losses.item()

            # Clip gradients to prevent exploding gradients:
            if params.gradient["clip"]:
                nn.utils.clip_grad_value_(
                    model.parameters(), gradient_threshold,
                )

            optimizer.step()
            del x, y, out, losses, equations

        torch.cuda.empty_cache()
        gc.collect()
        scheduler.step()

        if (ep + 1) % gradient_threshold_step_interval == 0:
            gradient_threshold /= 10.0
            logger.debug(f"{gradient_threshold=}")

        if ep % log_interval == 0:
            t2 = default_timer()
            epoch_train_time = t2 - t1
            avg_train_l2 = train_l2 / train_count
            print(
                f"Epoch {ep}: | "
                f"Time: {epoch_train_time:.2f}s | "
                f"Loss: {avg_train_l2:.4f}"
            )

            if params.wandb_log:
                values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
                wandb.log(values_to_log, step=ep, commit=True)

    model.eval()
    t1 = default_timer()
    test_l2 = 0.0
    n_test = 0
    _tqdm = tqdm.tqdm if script else tqdm.tqdm_notebook
    test_loader_tqdm = _tqdm(
        test_loader, desc=f"Final Exam ({epochs} / {epochs})", leave=False, ncols=100
    )
    with torch.no_grad():
        for x, y in test_loader_tqdm:
            equations = y[1]
            x = x[0].cuda()
            y = y[0].cuda()

            out, *_ = model(x, equations=equations)

            for k, eq in enumerate(equations):
                loss = loss_fn(y[k].view(1, -1), out[k].view(1, -1),)
                test_l2 += loss.item()
                n_test += 1

    test_l2 /= n_test
    t2 = default_timer()
    test_time = t2 - t1
    print(f"Time: {test_time:.2f}s\n" f"Loss: {test_l2:.6f}")

    if params.wandb_log:
        wandb.log({"test_error": test_l2}, commit=True)


def test_single_physics(
    model: nn.Module,
    test_loader: data.DataLoader,
    loss_fn,
    start: int,
    stop: int,
    script=True,
) -> None:
    """
    Test a single physics model on a given test dataset.

    Args:
        model (nn.Module): The physics model to be tested.
        test_loader (data.DataLoader): The test data loader.
        loss_fn: The loss function used for evaluation.
        start (int): The starting index of the test data.
        stop (int): The stopping index of the test data.
        script (bool, optional): Whether to use script mode for tqdm. Defaults to True.

    Returns:
        None
    """
    t1 = default_timer()
    test_l2 = 0.0
    n_test = 0

    with torch.no_grad():
        _trange = tqdm.trange if script else tqdm.tnrange
        test_loader_trange = _trange(
            start, stop, desc=f"Testing [{start} : {stop}]", leave=False,
        )

        for i in test_loader_trange:
            x, y = test_loader.dataset[i]
            eq = torch.Tensor([x[1]])
            x = x[0].unsqueeze(0).cuda()
            y = y[0].unsqueeze(0).cuda()

            out, *_ = model(x, equations=eq)
            loss = loss_fn(y[0].reshape(-1), out[0].reshape(-1),)
            test_l2 += loss.item()
            n_test += 1

    t2 = default_timer()
    test_time = t2 - t1
    print(f"Time: {test_time:.2f}s\n" f"Loss: {test_l2 / n_test:.6f}")
