from YParams import YParams
import wandb
import argparse
import torch
import numpy as np
from data_utils.data_loaders import *
from data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from layers.variable_encoding import *
from models.get_models import *
from train.trainer import nonuniform_mesh_trainer
from utils import *
from models.model_helpers import count_parameters
from test.evaluations import missing_variable_testing
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="?", default="base_config", type=str)
    parser.add_argument("--ntrain", nargs="?", default=None, type=int)
    parsed_args = parser.parse_args()

    config = parsed_args.config
    print("Loading config", config)
    params = YParams("./config/ssl_ns_elastic.yaml", config, print_params=True)

    if parsed_args.ntrain is not None:
        params.ntrain = parsed_args.ntrain
        print("Overriding ntrain to", params.ntrain)

    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    params.config = config
    # Set up WandB logging
    params.wandb_name = config
    params.wandb_group = params.nettype
    if params.wandb_log:
        wandb.login(key=get_wandb_api_key())
        wandb.init(
            config=params,
            name=params.wandb_name,
            group=params.wandb_group,
            project=params.wandb_project,
            entity=params.wandb_entity,
        )

    if params.pretrain_ssl:
        stage = StageEnum.RECONSTRUCTIVE
    else:
        stage = StageEnum.PREDICTIVE

    variable_encoder = None
    token_expander = None

    if params.nettype == "transformer":
        if params.grid_type == "uniform":
            encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(params)
        else:
            encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(
                params
            )

            if params.use_variable_encoding:
                variable_encoder = get_variable_encoder(params)
                k = variable_encoder(torch.randn(1317, 2), equation=["NS"])
                k = variable_encoder(torch.randn(1317, 2))
                token_expander = TokenExpansion(
                    sum([params.equation_dict[i] for i in params.equation_dict.keys()]),
                    params.n_encoding_channels,
                    params.n_static_channels,
                )

        print("Parameters Encoder", count_parameters(encoder), "x10^6")
        print("Parameters Decoder", count_parameters(decoder), "x10^6")
        print("Parameters Perdictor", count_parameters(predictor), "x10^6")

        model = SSLWrapper(
            params, encoder, decoder, contrastive, predictor, stage=stage
        )
        if params.grid_type != "uniform":
            print("Setting the Grid")
            mesh = get_mesh(params.input_mesh_location)
            input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()
            model.set_initial_mesh(input_mesh)

    elif params.nettype in ["simple", "gnn", "deeponet", "vit", "unet"]:
        model = get_model_fno(params)
        print("Parameters Model", count_parameters(model), "x10^6")
        input_mesh = None

    print(list(params.equation_dict.keys()))
    dataset = NsElasticDataset(
        params.data_location,
        equation=list(params.equation_dict.keys()),
        mesh_location=params.input_mesh_location,
        params=params,
    )

    train, test = dataset.get_dataloader(
        params.mu_list,
        params.dt,
        ntrain=params.get("ntrain"),
        ntest=params.get("ntest"),
        sample_per_inlet=params.sample_per_inlet,
    )

    normalizer = dataset.normalizer
    normalizer.cuda()

    if params.training_stage == "fine_tune":
        print(f"Loading Pretrained weights from {params.pretrain_weight}")
        model.encoder.load_state_dict(torch.load(params.pretrain_weight), strict=True)
        if params.use_variable_encoding:
            print(f"Loading Pretrained weights from {params.NS_variable_encoder_path}")
            if "NS" in params.equation_dict.keys():
                print("Loading NS variable encoder")
                variable_encoder.load_encoder("NS", params.NS_variable_encoder_path)
                if params.freeze_encoder:
                    variable_encoder.freeze("NS")

            if (
                "ES" in params.equation_dict.keys()
                and params.ES_variable_encoder_path is not None
            ):
                print("Loading ES variable encoder")
                variable_encoder.load_encoder("ES", params.ES_variable_encoder_path)
                if params.freeze_encoder:
                    variable_encoder.freeze("ES")
    model = model.cuda()
    if variable_encoder is not None:
        variable_encoder.cuda()
    if token_expander is not None:
        token_expander.cuda()
    nonuniform_mesh_trainer(
        model,
        train,
        test,
        params,
        wandb_log=params.wandb_log,
        log_test_interval=params.wandb_log_test_interval,
        normalizer=normalizer,
        stage=stage,
        variable_encoder=variable_encoder,
        token_expander=token_expander,
        initial_mesh=input_mesh,
    )

    if params.pretrain_ssl and not params.ssl_only:
        model.stage = StageEnum.PREDICTIVE
        nonuniform_mesh_trainer(
            model,
            train,
            test,
            params,
            wandb_log=params.wandb_log,
            log_test_interval=params.wandb_log_test_interval,
            normalizer=normalizer,
            stage=model.stage,
            variable_encoder=variable_encoder,
            token_expander=token_expander,
            initial_mesh=input_mesh,
        )

    grid_non, grid_uni = get_meshes(params.input_mesh_location, params.grid_size)

    test_augmenter = MaskerNonuniformMesh(
        grid_non_uni=grid_non.clone().detach(),
        gird_uni=grid_uni.clone().detach(),
        radius=params.masking_radius,
        drop_type=params.drop_type,
        drop_pix=params.drop_pix_val,
        channel_aug_rate=params.channel_per_val,
        channel_drop_rate=params.channel_drop_per_val,
        verbose=True,
    )

    missing_variable_testing(
        model,
        test,
        test_augmenter,
        normalizer,
        "sl",
        params,
        variable_encoder=variable_encoder,
        token_expander=token_expander,
        initial_mesh=input_mesh,
    )

    if params.wandb_log:
        wandb.finish()
