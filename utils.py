import datetime
import logging
import os
import pathlib
import psutil
import re
import signal

from typing import List

import h5py
# from haikunator import Haikunator
import numpy as np
import psutil
import torch
import torch.nn as nn

# HAIKU = haikunator.Haikunator()


def prepare_input(
        x,
        static_features,
        params,
        variable_encoder,
        token_expander,
        initial_mesh,
        data):
    if variable_encoder is not None and token_expander is not None:
        if params.grid_type == 'uniform':
            inp = token_expander(x, variable_encoder(x),
                                 static_features.cuda())
        elif params.grid_type == 'non uniform':
            initial_mesh = initial_mesh.cuda()
            equation = [i[0] for i in data['equation']]
            inp = token_expander(
                x,
                variable_encoder(
                    initial_mesh +
                    data['d_grid_x'].cuda()[0],
                    equation),
                static_features.cuda())
    elif params.n_static_channels > 0 and params.grid_type == 'non uniform':
        inp = torch.cat(
            [x, static_features[:, :, :params.n_static_channels].cuda()], dim=-1)
    else:
        inp = x
    return inp


def get_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()

def get_mesh(params):
    """Get the mesh from a location."""
    if hasattr(params, "text_mesh") and params.text_mesh:
        # load mesh_x and mesh_y from txt file as np array
        mesh_x = np.loadtxt(params.mesh_x)
        mesh_y = np.loadtxt(params.mesh_y)
        # create mesh from mesh_x and mesh_y
        mesh = np.zeros((mesh_x.shape[0], 2))
        mesh[:, 0] = mesh_x
        mesh[:, 1] = mesh_y
    else:
        h5f = h5py.File(params.input_mesh_location, 'r')
        mesh = h5f['mesh/coordinates']

    if params.super_resolution:
        # load mesh_x and mesh_y from txt file as np array
        mesh_x = np.loadtxt(params.super_resolution_mesh_x)
        mesh_y = np.loadtxt(params.super_resolution_mesh_y)
        # create mesh from mesh_x and mesh_y
        mesh_sup = np.zeros((mesh_x.shape[0], 2))
        mesh_sup[:, 0] = mesh_x
        mesh_sup[:, 1] = mesh_y
        # merge it with the original mesh
        mesh = np.concatenate((mesh, mesh_sup), axis=0)

        print("Super Resolution Mesh Shape", mesh.shape)

    if hasattr(
            params,
            'sub_sample_size') and params.sub_sample_size is not None:
        mesh_size = mesh.shape[0]
        indexs = [i for i in range(mesh_size)]
        np.random.seed(params.random_seed)
        sub_indexs = np.random.choice(
            indexs, params.sub_sample_size, replace=False)
        mesh = mesh[sub_indexs, :]

    return mesh[:]


# TODO add collision checks
# TODO add opts to toggle haiku and date fixes
def save_model(
        model,
        directory: pathlib.Path,
        stage=None,
        sep='_',
        name=None,
        config=None):
    """Saves a model with a unique prefix/suffix

    The model is prefixed with is date (formatted like YYMMDDHHmm)
    and suffixed with a "Heroku-like" name (for verbal reference).

    Params:
    ---
    stage: None | StageEnum
        Controls the infix of the model name according to the following mapping:
        | None -> "model"
        | RECONSTRUCTIVE -> "reconstructive"
        | PREDICTIVE -> "predictive"
    """
    prefix = datetime.datetime.utcnow().strftime("%y%m%d%H%M")
    infix = "model"
    if stage is not None:
        infix = stage.value.lower()
    # suffix = Haikunator.haikunate(token_length=0, delimiter=sep)

    torch.save(model.state_dict(), directory / f"{name}{sep}{config}{sep}.pth")


def extract_pids(message) -> List[int]:
    # Assume `message` has a preamble followed by a sequence of tokens like
    # "Process \d+" with extra characters in between such tokens.

    pattern = re.compile("(Process \\d+)")
    # Contains "Process" tokens and extra characters, interleaved:
    tokens = pattern.split(message)
    # print('\n'.join(map(repr, zip(split[1::2], split[2::2]))))

    pattern2 = re.compile("(\\d+)")
    # print('\n'.join([repr((s, pattern2.search(t)[0])) for t in tokens[1::2]]))
    pids = [int(pattern2.search(t)[0]) for t in tokens[1::2]]

    return pids


# https://psutil.readthedocs.io/en/latest/#kill-process-tree
def signal_process_tree(
    pid,
    sig=signal.SIGTERM,
    include_parent=True,
    timeout=None,
    on_terminate=None,
    logger=None,
):
    """Kill a process tree (including grandchildren) with signal ``sig``

    Return a (gone, still_alive) tuple.

    Parameters
    ---
    timeout: float
        Time, in seconds, to wait on each signaled process.
    on_terminate: Optional[Callable]
        A callback function which is called as soon as a child terminates.
        Optional.
    """
    assert pid != os.getpid(), "won't kill myself"
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    if logger is None:
        logger = logging.getLogger()

    wait_children = []
    for p in children:
        try:
            p.send_signal(sig)
            wait_children.append(p)
        except psutil.AccessDenied:
            logger.error(f"Unable to terminate Process {pid} (AccessDenied)")
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(
        wait_children,
        timeout=timeout,
        callback=on_terminate,
    )
    return (gone, alive)


def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel()
         for p in model.parameters()]
    )


def signal_my_processes(
    username,
    pids,
    sig=signal.SIGTERM,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger()
    my_pids = []
    for pid in pids:
        p = psutil.Process(pid)
        with p.oneshot():
            p = p.as_dict(attrs=["username", "status"])

        # TODO add other states to the filter
        if p["username"] == username and p["status"] == "sleeping":
            my_pids.append(pid)
        else:
            _p = {"pid": pid, **p}
            logger.warning(f"Cannot signal process: {_p}")

    for my_pid in my_pids:
        gone, alive = signal_process_tree(pid, sig, timeout=60, logger=logger)
        logger.info(f"{gone=}, {alive=}")
