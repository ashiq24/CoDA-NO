import torch
from data_utils.data_utils import *
import torch.nn as nn
from timeit import default_timer
from models.get_models import *
import gc
from tqdm import tqdm
import wandb


def missing_variable_testing(
        model,
        test_loader,
        augmenter,
        normalizer,
        stage,
        params,
        variable_encoder=None,
        token_expander=None,
        initial_mesh=None,):
    """
    This function is used to test the model on missing/maksed variable testing
    Masking or dropping the variable is done by augmenter
    """
    print('Evaluating for Stage: ', stage)
    with torch.no_grad():
        ntest = 0
        test_l2 = 0
        loss_p = nn.MSELoss()
        for data in test_loader:
            x, y = data['x'], data['y']

            static_features = data['static_features']
            equation = [i[0] for i in data['equation']]
            # print(x.shape, y.shape, static_features.shape, data['equation'])
            x, y = x.cuda(), y.cuda()

            if augmenter is not None:
                x, _ = batched_masker(x, augmenter)

            # print(initial_mesh.shape, data['d_grid_x'].cuda()[0].shape, equation)

            if variable_encoder is not None and token_expander is not None:
                inp = token_expander(x, variable_encoder(
                    initial_mesh + data['d_grid_x'].cuda()[0], equation), static_features.cuda())
            elif params.n_static_channels > 0:
                inp = torch.cat(
                    [x, static_features[:, :, :params.n_static_channels].cuda()], dim=-1)
            else:
                inp = x

            if params.grid_type == "non uniform":
                '''
                Assume non uniform grids requires
                updating grid for every sample. We need to
                suppy the grid.

                last 3 channel is displacement, taking (x,y), z is 0
                '''
                with torch.no_grad():
                    if stage == StageEnum.RECONSTRUCTIVE:
                        out_grid_displacement = data['d_grid_x'].cuda()[0]
                        in_grid_displacement = data['d_grid_x'].cuda()[0]
                    else:
                        out_grid_displacement = data['d_grid_y'].cuda()[0]
                        in_grid_displacement = data['d_grid_x'].cuda()[0]
            else:
                print("Uniform Grids")
                out_grid_displacement = None
                in_grid_displacement = None

            batch_size = x.shape[0]
            # print(in_grid_displacement, out_grid_displacement)
            out = model(inp, out_grid_displacement=out_grid_displacement,
                        in_grid_displacement=in_grid_displacement)

            if isinstance(out, (list, tuple)):
                out = out[0]
            ntest += 1

            target = y.clone()

            test_l2 += loss_p(target.reshape(batch_size, -1), out.reshape(batch_size, -1)
                              ).item()

    test_l2 /= ntest
    t2 = default_timer()

    # XXX Consolidat all W&B logs in one file.
    if wandb_log:
        wandb.log({'Augmented test_error_' + stage: test_l2}, commit=True)
    print(f"Augmented Test Error  {stage}: ", test_l2)
