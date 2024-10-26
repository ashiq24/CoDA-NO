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
from utils import *

def get_grid_displacement(params, stage, data):
    if params.grid_type == "non uniform":
        with torch.no_grad():
            if stage == StageEnum.RECONSTRUCTIVE:
                out_grid_displacement = data['d_grid_x'].cuda()[0]
                in_grid_displacement = data['d_grid_x'].cuda()[0]
            else:
                out_grid_displacement = data['d_grid_y'].cuda()[0]
                in_grid_displacement = data['d_grid_x'].cuda()[0]
    else:
        out_grid_displacement = None
        in_grid_displacement = None
    return out_grid_displacement, in_grid_displacement

def trainer(
    model,
    train_loader,
    test_loader,
    params,
    wandb_log=False,
    log_test_interval=1,
    stage=StageEnum.RECONSTRUCTIVE,
    variable_encoder=None,
    token_expander=None,
    initial_mesh=None,
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
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, patience=scheduler_step, factor=scheduler_gamma)

    loss_p = nn.MSELoss(reduction='sum')

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_count = 0
        train_loader_iter = tqdm.tqdm(
            train_loader, desc=f'Epoch {ep}/{epochs}', leave=False, ncols=100)

        for data in train_loader_iter:
            optimizer.zero_grad()
            x, y = data['x'].cuda(), data['y'].cuda()
            static_features = data['static_features']
            if stage == StageEnum.RECONSTRUCTIVE and params.masking:
                x = model.do_mask(x)

            inp = prepare_input(
                x,
                static_features,
                params,
                variable_encoder,
                token_expander,
                initial_mesh,
                data)
            batch_size = x.shape[0]
            if params.grid_type == "non uniform":
                out_grid_displacement, in_grid_displacement = get_grid_displacement(
                    params, stage, data)
            elif params.grid_type == "uniform":
                out_grid_displacement = None
                in_grid_displacement = None

            out = model(inp, out_grid_displacement=out_grid_displacement,
                        in_grid_displacement=in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]
            if getattr(params, 'horizontal_skip', False):
                out = out + x

            train_count += 1
            target = x.clone() if stage == StageEnum.RECONSTRUCTIVE else y.clone()
            loss = loss_p(target.reshape(
                batch_size, -1), out.reshape(batch_size, -1)) / (x.shape[0] * x.shape[-1] * x.shape[-2])
            loss.backward()

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
            scheduler.step()

        t2 = default_timer()
        epoch_train_time = t2 - t1

        if ep % log_test_interval == 0:
            values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
            print(
                f"Epoch {ep}: Time: {epoch_train_time:.3f}s, Loss: {avg_train_l2:.7f}")
            if wandb_log:
                wandb.log(values_to_log, commit=True)

        if ep % params.weight_saving_interval == 0 or ep == epochs - 1:
            stage_string = 'ssl' if stage == StageEnum.RECONSTRUCTIVE else 'sl'
            if params.nettype != 'transformer':
                torch.save(model.state_dict(), weight_path +
                           params.config + "_" + str(ep) + '.pt')
            else:
                weight_path_model_encoder = weight_path + params.config + \
                    "_" + stage_string + '_encoder_' + str(ep) + '.pt'
                weight_path_model_decoder = weight_path + params.config + \
                    "_" + stage_string + '_decoder_' + str(ep) + '.pt'
                weight_path_whole_model = weight_path + params.config + \
                    "_" + stage_string + '_whole_model_' + str(ep) + '.pt'
                torch.save(model.encoder.state_dict(),
                           weight_path_model_encoder)
                torch.save(model.decoder.state_dict(),
                           weight_path_model_decoder)
                torch.save(model.state_dict(), weight_path_whole_model)
                if variable_encoder is not None:
                    variable_path = weight_path + params.config + \
                        "_variable_encoder_" + str(ep)
                    variable_encoder.save_all_encoder(variable_path)

    model.eval()
    test_l2 = 0.0
    ntest = 0
    loss_p = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for data in test_loader:
            x, y = data['x'].cuda(), data['y'].cuda()
            static_features = data['static_features']
            inp = prepare_input(
                x,
                static_features,
                params,
                variable_encoder,
                token_expander,
                initial_mesh,
                data)
            out_grid_displacement, in_grid_displacement = get_grid_displacement(
                params, stage, data)
            batch_size = x.shape[0]
            out = model(inp, in_grid_displacement=in_grid_displacement,
                        out_grid_displacement=out_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]
            if getattr(params, 'horizontal_skip', False):
                out = out + x

            ntest += x.shape[0]
            target = x.clone() if stage == StageEnum.RECONSTRUCTIVE else y.clone()
            test_l2 += loss_p(target.reshape(batch_size, -1),
                              out.reshape(batch_size, -1)).item()

    test_l2 /= (ntest * x.shape[-1] * x.shape[-2])
    t2 = default_timer()

    if wandb_log:
        stage_string = 'ssl' if stage == StageEnum.RECONSTRUCTIVE else 'sl'
        wandb.log({'test_error_' + stage_string: test_l2}, commit=True)
    print("Test Error : " + stage_string, test_l2)
