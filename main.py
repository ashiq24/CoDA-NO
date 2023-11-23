from YParams import YParams
import os
import wandb
import sys
import torch
from data_utils.data_loaders import *
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from data_utils.data_utils import MakserNonuniform, batched_masker, MaskerUniform, get_meshes
from models.codano import CodANO
from models.get_models import *
from train.trainer import simple_trainer
from utils import get_wandb_api_key
from models.model_helpers import count_parameters
from test.evaluations import missing_variable_testing

import random

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    config = sys.argv[1]
    print("Loading config", config)
    params = YParams('./config/ssl_ns_elastic.yaml', config, print_params=True)

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

    if params.pretrain_ssl:
        stage = 'ssl'
    else:
        stage = 'sl'

    if params.nettype == 'transformer':
        if params.grid_type == 'uniform':
            encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(
                params)
        else:
            encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(
                params)

        print("Parameters Encoder", count_parameters(encoder),"x10^6")
        print("Parameters Decoder", count_parameters(decoder),"x10^6")
        print("Parameters Perdictor", count_parameters(predictor),"x10^6")
        # if params.grid_type == 'uniform':
        model = SslWrapper(
            params,
            encoder,
            decoder,
            contrastive,
            predictor,
            stage=stage)
        if params.grid_type != 'uniform':
            print("Setting the Grid")
            mesh = np.loadtxt(params.input_mesh_location, delimiter=',')
            input_mesh = torch.transpose(torch.stack([torch.tensor(
                mesh[0, :]), torch.tensor(mesh[1, :])]), 0, 1).type(torch.float).cuda()
            model.set_initial_mesh(input_mesh)
    elif params.nettype == 'simple':
        model = get_model_fno(params)
        print("Parameters Model", count_parameters(model),"x10^6")

    model = model.cuda()
    # non-uniform dataset
    dataset = NsElasticDataset(params.data_location)
    # train, test = dataset.get_onestep_dataloader(location=params.data_location, dt=params.dt, ntrain=params.get('ntrain'),
    #                                              ntest=params.get('ntest'))

    train, test = dataset.get_dataloader(params.mu_list,params.dt,ntrain=params.get('ntrain'),ntest=params.get('ntest'))

    normalizer = dataset.normalizer
    normalizer.cuda()

    # uniform dataset dummy
    # train, test = get_dummy_dataloaders()
    if params.training_stage == 'fine_tune':
        print(f"Loading Pretrained weights from {params.pretrain_weight}")
        model.load_state_dict(torch.load(params.pretrain_weight))

    simple_trainer(
        model,
        train, 
        test,
        params,
        wandb_log=params.wandb_log,
        log_test_interval=params.wandb_log_test_interval,
        normalizer=normalizer,
        stage=stage)

    if params.pretrain_ssl:
        # if we were pre-training (ssl), then we will train (sl)
        model.stage = 'sl'
        simple_trainer(
            model,
            train,
            test,
            params,
            wandb_log=params.wandb_log,
            log_test_interval=params.wandb_log_test_interval,
            normalizer=normalizer,
            stage=model.stage)

    grid_non, grid_uni = get_meshes(
        params.input_mesh_location, params.grid_size)

    test_augmenter = MakserNonuniform(
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
        normalizer,
        'sl',
        params)

    if params.wandb_log:
        wandb.finish()
