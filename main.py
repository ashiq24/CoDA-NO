from YParams import YParams
import wandb
import argparse
import torch
import numpy as np
from data_utils.data_loaders import *
from data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from layers.variable_encoding import *
from models.get_models import *
from train.trainer import trainer
from utils import *
from models.model_helpers import count_parameters
from test.evaluations import missing_variable_testing
import random


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="?", default="FSI", type=str)
    parser.add_argument("--config", nargs="?", default="base_config", type=str)
    parser.add_argument("--ntrain", nargs="?", default=None, type=int)
    parser.add_argument("--epochs", nargs="?", default=None, type=int)
    parser.add_argument("--random_seed", nargs="?", default=42, type=int)
    parser.add_argument("--scheduler_step", nargs="?", default=None, type=int)
    parser.add_argument("--batch_size", nargs="?", default=None, type=int)
    parsed_args = parser.parse_args()

    if parsed_args.exp == "FSI":
        config_file = './config/ssl_ns_elastic.yaml'
    elif parsed_args.exp == "RB":
        config_file = './config/RB_config.yaml'
    else:
        raise ValueError("Unknown experiment type")

    config = parsed_args.config
    print("Loading config", config)
    params = YParams(config_file, config, print_params=True)

    if parsed_args.ntrain is not None:
        params.ntrain = parsed_args.ntrain
        print("Overriding ntrain to", params.ntrain)
    if parsed_args.random_seed is not None:
        params.random_seed = parsed_args.random_seed
        print("Overriding random seed to", params.random_seed)
    if parsed_args.epochs is not None:
        params.epochs = parsed_args.epochs
        print("Overriding epochs to", params.epochs)
    if parsed_args.scheduler_step is not None:
        params.scheduler_step = parsed_args.scheduler_step
        print("Overriding scheduler step to", params.scheduler_step)
    if parsed_args.batch_size is not None:
        params.batch_size = parsed_args.batch_size
        print("Overriding batch size to", params.batch_size)

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
            entity=params.wandb_entity)

    # stage of training: reconstructive (Self supervised traning)
    # or predictive (Supervised training either finetuning or from scratch)
    if params.pretrain_ssl:
        stage = StageEnum.RECONSTRUCTIVE
    else:
        stage = StageEnum.PREDICTIVE

    variable_encoder = None
    token_expander = None

    if params.nettype == 'transformer':
        if params.grid_type == 'uniform':
            encoder, decoder, contrastive, predictor = get_ssl_models_codano(
                params)
            input_mesh = None
        else:
            encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(
                params)

        if params.use_variable_encoding:
            variable_encoder = get_variable_encoder(params)
            token_expander = TokenExpansion(sum([params.equation_dict[i] for i in params.equation_dict.keys(
            )]), params.n_encoding_channels, params.n_static_channels, params.grid_type == 'uniform')

        print("Parameters Encoder", count_parameters(encoder), "x10^6")
        print("Parameters Decoder", count_parameters(decoder), "x10^6")
        print("Parameters Perdictor", count_parameters(predictor), "x10^6")
        if params.wandb_log:
            wandb.log(
                {'Encoder #parameters': count_parameters(encoder)}, commit=True)
            wandb.log(
                {'Decoder #parameters': count_parameters(decoder)}, commit=True)
            wandb.log(
                {'Predictor #parameters': count_parameters(predictor)}, commit=True)

        model = SSLWrapper(
            params,
            encoder,
            decoder,
            contrastive,
            predictor,
            stage=stage)

        if params.grid_type != 'uniform':
            print("Setting the Grid")
            mesh = get_mesh(params)
            input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()
            model.set_initial_mesh(input_mesh)

    elif params.nettype in ['simple', 'gnn', 'deeponet', 'vit', 'unet', 'fno']:
        model = get_baseline_model(params)
        print("Parameters Model", count_parameters(model), "x10^6")
        wandb.log({'Model #parameters': count_parameters(model)}, commit=True)
        input_mesh = None

    print("PDE list", *list(params.equation_dict.keys()))

    if parsed_args.exp == 'FSI':
        # loading Fluid Stucture Interaction dataset
        dataset = NsElasticDataset(
            params.data_location,
            equation=list(params.equation_dict.keys()),
            mesh_location=params.input_mesh_location,
            params=params)
        train, test = dataset.get_dataloader(params.mu_list, params.dt, ntrain=params.get(
            'ntrain'), ntest=params.get('ntest'), sample_per_inlet=params.sample_per_inlet)
    elif parsed_args.exp == 'RB':
        # loading Rayleigh-Benard dataset
        train, test = get_RB_dataloader(params)

    if getattr(params, 'evaluate_only', False):
        # setting satge to predictive for evaluation
        # load model weights
        stage = StageEnum.PREDICTIVE
        model.load_state_dict(torch.load(params.model_path), strict=False)
        if params.nettype == 'transformer' and params.use_variable_encoding:
            if "NS" in params.equation_dict.keys():
                print("Loading NS variable encoder")
                variable_encoder.load_encoder(
                    "NS", params.NS_variable_encoder_path)
            if "ES" in params.equation_dict.keys() and params.ES_variable_encoder_path is not None:
                print("Loading ES variable encoder")
                variable_encoder.load_encoder(
                    "ES", params.ES_variable_encoder_path)
            if "T" in params.equation_dict.keys() and params.T_variable_encoder_path is not None:
                print("Loading T variable encoder")
                variable_encoder.load_encoder(
                    "T", params.T_variable_encoder_path)
                if params.freeze_encoder:
                    variable_encoder.freeze("T")

    elif params.training_stage == 'fine_tune':
        # load only encooder and vaariable encoder weights (VSPE)
        print(f"Loading Pretrained weights from {params.pretrain_weight}")
        model.encoder.load_state_dict(torch.load(
            params.pretrain_weight), strict=True)
        if params.use_variable_encoding:
            print(
                f"Loading Pretrained weights from {params.NS_variable_encoder_path}")

            if "NS" in params.equation_dict.keys():
                print("Loading NS variable encoder")
                variable_encoder.load_encoder(
                    "NS", params.NS_variable_encoder_path)
                if params.freeze_encoder:
                    variable_encoder.freeze("NS")

            if "ES" in params.equation_dict.keys() and params.ES_variable_encoder_path is not None:
                print("Loading ES variable encoder")
                variable_encoder.load_encoder(
                    "ES", params.ES_variable_encoder_path)
                if params.freeze_encoder:
                    variable_encoder.freeze("ES")

            if "T" in params.equation_dict.keys() and params.T_variable_encoder_path is not None:
                print("Loading T variable encoder")
                variable_encoder.load_encoder(
                    "T", params.T_variable_encoder_path)
                if params.freeze_encoder:
                    variable_encoder.freeze("T")

    # Move model and encoders to GPU
    model = model.cuda()
    if variable_encoder is not None:
        variable_encoder.cuda()
    if token_expander is not None:
        token_expander.cuda()

    if not getattr(params, 'evaluate_only', False):
        # Train the model
        trainer(
            model,
            train,
            test,
            params,
            wandb_log=params.wandb_log,
            log_test_interval=params.wandb_log_test_interval,
            stage=stage,
            variable_encoder=variable_encoder,
            token_expander=token_expander,
            initial_mesh=input_mesh)

        if getattr(params, 'missing_var_test', False):
            # evaluate on missing variables and partially observed
            # variabless

            grid_non, grid_uni = get_meshes(
                params, params.grid_size)
            test_augmenter = MaskerNonuniformMesh(
                grid_non_uni=grid_non.clone().detach(),
                gird_uni=grid_uni.clone().detach(),
                radius=params.masking_radius,
                drop_type=params.drop_type,
                drop_pix=params.drop_pix_val,
                channel_aug_rate=params.channel_per_val,
                channel_drop_rate=params.channel_drop_per_val,
                verbose=True)
            missing_variable_testing(
                model,
                test,
                test_augmenter,
                'sl',
                params,
                variable_encoder=variable_encoder,
                token_expander=token_expander,
                initial_mesh=input_mesh)
    else:
        # Evaluate the model on the
        # unaugmentted the test set

        missing_variable_testing(
            model,
            test,
            augmenter=None,
            normalizer=None,
            stage=stage,
            params=params,
            variable_encoder=variable_encoder,
            token_expander=token_expander,
            initial_mesh=input_mesh)

    if params.wandb_log:
        wandb.finish()
