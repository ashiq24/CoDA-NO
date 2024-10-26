import torch
from data_utils.data_utils import *
import torch.nn as nn
from timeit import default_timer
from models.get_models import *
from tqdm import tqdm
import wandb
from utils import *
from train.trainer import *


def missing_variable_testing(
        model,
        test_loader,
        augmenter=None,
        stage=StageEnum.PREDICTIVE,
        params=None,
        variable_encoder=None,
        token_expander=None,
        initial_mesh=None,
        wandb_log=False):
    print('Evaluating for Stage: ', stage)
    model.eval()
    with torch.no_grad():
        ntest = 0
        test_l2 = 0
        test_l1 = 0
        loss_p = nn.MSELoss()
        loss_l1 = nn.L1Loss()
        t1 = default_timer()
        predictions = []
        for data in test_loader:
            x, y = data['x'].cuda(), data['y'].cuda()
            static_features = data['static_features']

            if augmenter is not None:
                x, _ = batched_masker(x, augmenter)

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
            out = model(inp, out_grid_displacement=out_grid_displacement,
                        in_grid_displacement=in_grid_displacement)

            if getattr(params, 'horizontal_skip', False):
                out = out + x

            if isinstance(out, (list, tuple)):
                out = out[0]

            ntest += 1
            target = y.clone()

            predictions.append((out, target))

            test_l2 += loss_p(target.reshape(batch_size, -1),
                              out.reshape(batch_size, -1)).item()
            test_l1 += loss_l1(target.reshape(batch_size, -1),
                               out.reshape(batch_size, -1)).item()

    test_l2 /= ntest
    test_l1 /= ntest
    t2 = default_timer()
    avg_time = (t2 - t1) / ntest

    wandb.log({'Augmented test_error_l2': test_l2}, commit=True)
    wandb.log({'Augmented test_error_l1': test_l1}, commit=True)
    wandb.log({'Avg test_time': avg_time}, commit=True)
    print(f"Augmented Test Error  {stage}: ", test_l2)

    if hasattr(params, 'save_predictions') and params.save_predictions:
        torch.save(predictions[:50], f'../xy/predictions_{params.config}.pt')
