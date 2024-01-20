import datetime
import logging
import os
import pathlib
import psutil
import re
import signal

from typing import List

import h5py
#import haikunator
import numpy as np
import psutil
import torch
import torch.nn as nn

#HAIKU = haikunator.Haikunator()

def get_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


class TokenExpansion(nn.Module):
    def __init__(self, n_variables: int, n_encoding_channels, n_static_channels: int) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.n_encoding_channels = n_encoding_channels
        self.n_static_channels = n_static_channels

        expansion_factor = 1 + self.n_static_channels + self.n_encoding_channels

        self.variable_channels = [
            i * expansion_factor for i in range(n_variables)]
        self.static_channels = []
        if self.n_static_channels != 0:
            for v in self.variable_channels:
                self.static_channels.extend(
                    range(v + 1, v + self.n_static_channels + 1))
        self.encoding_channels = []
        if self.n_encoding_channels != 0:
            self.encoding_channels = sorted(list(
                set(range(n_variables * expansion_factor))
                - set(self.variable_channels)
                - set(self.static_channels)
            ))

        print(self.variable_channels)
        print(self.static_channels)
        print(self.encoding_channels)

    def forward(self, inp: torch.Tensor, variable_encodings: torch.tensor, static_channels: torch.tensor) -> torch.Tensor:
        """
        x: (batch_size, n_variables)
        """
        x = torch.zeros((inp.shape[0], inp.shape[1], len(
            self.variable_channels) + len(self.encoding_channels) + len(self.static_channels)), device=inp.device, dtype=inp.dtype)
        x[:, :, self.variable_channels] = inp
        if self.n_static_channels != 0:
            # print(x[:, :, self.static_channels].shape, static_channels.repeat(
            #     x.shape[0], 1, 1).shape)
            x[:, :, self.static_channels] = static_channels.repeat(
                x.shape[0], 1, 1)
        if self.n_encoding_channels != 0:
            x[:, :, self.encoding_channels] = variable_encodings.repeat(
                
                x.shape[0], 1, 1)

        return x


def get_mesh(location):
    """Get the mesh from a location."""
    h5f = h5py.File(location, 'r')
    mesh = h5f['mesh/coordinates']
    return mesh[:]


# TODO add collision checks
# TODO add opts to toggle haiku and date fixes
def save_model(model, directory: pathlib.Path, stage=None, sep='_'):
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
    suffix = HAIKU.haikunate(token_length=0, delimiter=sep)

    torch.save(model.state_dict(), directory / f"{prefix}{sep}{infix}{sep}{suffix}.pth")


def extract_pids(message) -> List[int]:
    # Assume `message` has a preamble followed by a sequence of tokens like
    # "Process \d+" with extra characters in between such tokens.
    
    pattern = re.compile("(Process \d+)")
    # Contains "Process" tokens and extra characters, interleaved:
    tokens = pattern.split(message)
    # print('\n'.join(map(repr, zip(split[1::2], split[2::2]))))
    
    pattern2 = re.compile("(\d+)")
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
