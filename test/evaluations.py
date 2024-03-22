import torch
from data_utils.data_utils import *
import torch.nn as nn
from timeit import default_timer
from models.get_models import *
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
    initial_mesh=None,
):
    """
    This function is used to test the model on missing/masked variable testing.
    The function takes in various parameters including the model, test_loader, augmenter, normalizer, stage, params, variable_encoder, token_expander, and initial_mesh.
    
    Parameters:
    - model: The model to be tested.
    - test_loader: The data loader for the test dataset.
    - augmenter: The augmenter used for masking or dropping variables.
    - normalizer: The normalizer used for normalizing the data.
    - stage: The stage of evaluation.
    - params: The parameters for the evaluation.
    - variable_encoder: The variable encoder used for encoding variables.
    - token_expander: The token expander used for expanding tokens.
    - initial_mesh: The initial mesh for the evaluation.
    
    Returns:
    - None
    
    This function evaluates the model on the test dataset by iterating through the test_loader and calculating the test error.
    It performs masking or dropping of variables using the augmenter and applies normalization using the normalizer.
    The evaluation is performed for a specific stage and the test error is calculated using the mean squared error loss.
    The test error is then logged and printed.
    """
    print("Evaluating for Stage: ", stage)
    with torch.no_grad():
        ntest = 0
        test_l2 = 0
        loss_p = nn.MSELoss()
        for data in test_loader:
            x, y = data["x"], data["y"]

            static_features = data["static_features"]
            equation = [i[0] for i in data["equation"]]
            x, y = x.cuda(), y.cuda()

            if augmenter is not None:
                x, _ = batched_masker(x, augmenter)

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
                supply the grid.

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
                print("Uniform Grids")
                out_grid_displacement = None
                in_grid_displacement = None

            batch_size = x.shape[0]
            out = model(
                inp,
                out_grid_displacement=out_grid_displacement,
                in_grid_displacement=in_grid_displacement,
            )

            if isinstance(out, (list, tuple)):
                out = out[0]
            ntest += 1

            target = y.clone()

            test_l2 += loss_p(
                target.reshape(batch_size, -1), out.reshape(batch_size, -1)
            ).item()

    test_l2 /= ntest
    t2 = default_timer()

    if params.wandb_log:
        wandb.log({"Augmented test_error_" + stage: test_l2}, commit=True)
    print(f"Augmented Test Error  {stage}: ", test_l2)
