# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tarfile
import urllib.request
import h5py
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from modulus.models.rnn.rnn_one2many import One2ManyRNN
import torch.nn.functional as F
from typing import Union
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path
from pyevtk.hl import imageToVTK

from deephyper.evaluator import profile, RunningJob
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob

from omegaconf import OmegaConf
import sys
import pandas as pd

import gc

# set device as GPU
device = "cuda"

def get_optimizer(name):
    if name == "Adadelta":
        optimizer = torch.optim.Adadelta
    elif name == "Adagrad":
        optimizer = torch.optim.Adagrad
    elif name == "Adam":
        optimizer = torch.optim.Adam
    elif name == "AdaBound":
        optimizer = torch.optim.AdaBound
    elif name == "RMSprop":
        optimizer = torch.optim.RMSprop
    elif name == "SGD":
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {name} does not exist.")
    return optimizer

class Logger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = {}

    def log(self, key, info):
        if key in self.logger.keys():
            self.logger[key] = np.append(self.logger[key],info)#.append(info)
        else:
            self.logger[key] = [info]

    def save(self, name):
        with open(self.path + f"log_{name}.pkl", "wb") as f:
            pickle.dump(self.logger, f)

def prepare_data(
    input_data_path,
    output_data_path,
    predict_nr_tsteps,
    start_timestep,
):
    """Data pre-processing"""
    if Path(output_data_path).is_file():
        pass
    else:
        data = h5py.File(input_data_path)
        list_data = []
        for i in range(len(list(data.keys()))):
            data_u = data[str(i)]["u"]
            data_v = data[str(i)]["v"]
            data_uv = np.stack([data_u, data_v], axis=0)
            data_uv = np.array(data_uv)
            list_data.append(data_uv)

        data.close()
        data_combined = np.stack(list_data, axis=0)

        h = h5py.File(output_data_path, "w")
        h.create_dataset(
            "invar",
            data=np.expand_dims(data_combined[:, :, start_timestep, ...], axis=2),
        )
        h.create_dataset(
            "outvar",
            data=data_combined[
                :, :, start_timestep + 1 : start_timestep + 1 + predict_nr_tsteps, ...
            ],
        )
        h.close()


def validation_step(model, dataloader):
    """Validation Step"""
    model.eval()
    valLoss = 0
    for data in dataloader:
        invar, outvar = data
        predvar = model(invar)
        loss = F.mse_loss(outvar, predvar)
        valLoss += loss.detach().cpu().numpy()

    # convert data to numpy
    outvar = outvar.detach().cpu().numpy()
    predvar = predvar.detach().cpu().numpy()

    # plotting
    for t in range(outvar.shape[2]):
        cellData = {
            "outvar_chan0": outvar[0, 0, t, ...],
            "outvar_chan1": outvar[0, 1, t, ...],
            "predvar_chan0": predvar[0, 0, t, ...],
            "predvar_chan1": predvar[0, 1, t, ...],
        }
        imageToVTK(f"./test/test_{t}", cellData=cellData)
    return valLoss / len(dataloader) 

class HDF5MapStyleDataset(Dataset):
    """Simple map-stype HDF5 dataset"""

    def __init__(
        self,
        file_path,
        device: Union[str, torch.device] = "cuda",
    ):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.keys[0]])

    def __getitem__(self, idx):
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                data[key] = np.array(f[key][idx])

        invar = torch.from_numpy(data["invar"])
        outvar = torch.from_numpy(data["outvar"])
        if self.device.type == "cuda":
            # Move tensors to GPU
            invar = invar.cuda()
            outvar = outvar.cuda()

        return invar, outvar

def run(job: RunningJob):
    cfg = OmegaConf.load('conf/config_3d.yaml')
    # Data download
    raw_train_data_path = to_absolute_path("./datasets/grayscott_training.hdf5")
    raw_test_data_path = to_absolute_path("./datasets/grayscott_test.hdf5")

    # Download data
    if Path(raw_train_data_path).is_file():
        pass
    else:
        print("Data download starting...")
        url = "https://zenodo.org/record/5148524/files/grayscott_training.tar.gz"
        os.makedirs(to_absolute_path("./datasets/"), exist_ok=True)
        output_path = to_absolute_path("./datasets/grayscott_training.tar.gz")
        urllib.request.urlretrieve(url, output_path)
        print("Data downloaded.")
        print("Extracting data...")
        with tarfile.open(output_path, "r") as tar_ref:
            tar_ref.extractall(to_absolute_path("./datasets/"))
        print("Data extracted")

    if Path(raw_test_data_path).is_file():
        pass
    else:
        print("Data download starting...")
        url = "https://zenodo.org/record/5148524/files/grayscott_test.tar.gz"
        os.makedirs(to_absolute_path("./datasets/"), exist_ok=True)
        output_path = to_absolute_path("./datasets/grayscott_test.tar.gz")
        urllib.request.urlretrieve(url, output_path)
        print("Data downloaded.")
        print("Extracting data...")
        with tarfile.open(output_path, "r") as tar_ref:
            tar_ref.extractall(to_absolute_path("./datasets/"))
        print("Data extracted")

    # Data pre-processing
    nr_tsteps_to_predict = 64
    nr_tsteps_to_test = 64
    start_timestep = 5

    train_save_path = "./train_data_gray_scott_one2many.hdf5"
    test_save_path = "./test_data_gray_scott_one2many.hdf5"

    # prepare data
    prepare_data(
        raw_train_data_path, train_save_path, nr_tsteps_to_predict, start_timestep
    )
    prepare_data(
        raw_test_data_path,
        test_save_path,
        nr_tsteps_to_test,
        start_timestep,
    )

    param = job.parameters.copy()
    train_dataset = HDF5MapStyleDataset(train_save_path, device="cuda")
    train_dataloader = DataLoader(
        train_dataset, batch_size=param["batch_size"], shuffle=True
    )
    test_dataset = HDF5MapStyleDataset(test_save_path, device="cuda")
    test_dataloader = DataLoader(
        test_dataset, batch_size=param["batch_size"], shuffle=False
    )

    # instantiate model
    arch = One2ManyRNN(
        input_channels=2, # fixed by tasks
        dimension=3, # fixed by tasks
        nr_tsteps=nr_tsteps_to_predict, # Time steps to predict fixed by tasks 
        nr_downsamples=param["nr_downsamples"],
        nr_residual_blocks=param["nr_residual_blocks"],
        nr_latent_channels=param["nr_latent_channels"],
    )

    if device == "cuda":
        arch.cuda()
    
    optimizer = get_optimizer(param["optimizer"])
    optimizer = optimizer(
        arch.parameters(), lr=param["lr"] 
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=param["lr_scheduler_gamma"]
    )

    #loaded_epoch = load_checkpoint(
    #    "./checkpoints",
    #    models=arch,
    #    optimizer=optimizer,
    #    scheduler=scheduler,
    #    device="cuda",
    #)

    logger = Logger("./log/")
    logger.log(
        "NumTrainableParams",
        sum(p.numel() for p in arch.parameters() if p.requires_grad),
    )
    # Training loop
    trainLoss = 0
    print("training num_mini_batch: {}, testing num_mini_batch: {}".format(len(train_dataloader), len(test_dataloader)))
    log_freq = int(len(train_dataloader)*0.1)
    for epoch in range(cfg.max_epochs + 1):
        # wrap epoch in launch logger for console logs
        num_mini_batch=len(train_dataloader),
        # go through the full dataset
        for i, data in enumerate(train_dataloader):
            invar, outvar = data
            optimizer.zero_grad()
            outpred = arch(invar)

            loss = F.mse_loss(outvar, outpred)
            loss.backward()
            optimizer.step()
            scheduler.step()
            trainLoss += loss.detach().cpu().numpy()
            if (i%log_freq == 0):   
                print("batch {}, loss = {}".format(i, loss.detach().cpu().numpy())) 
                sys.stdout.flush()
        trainLoss = trainLoss / num_mini_batch 

        valLoss = validation_step(arch, test_dataloader)
        print("Epoch: ", epoch)
        print("trainingLoss: ", trainLoss)
        print("valLoss: ", valLoss)
        sys.stdout.flush()

        logger.log("Epoch", epoch)
        logger.log("Learning Rate", optimizer.param_groups[0]["lr"])
        logger.log("TrainLoss", trainLoss)
        logger.log("ValLoss", valLoss)

        #if epoch % cfg.checkpoint_save_freq == 0:
        #    save_checkpoint(
        #        "./checkpoints",
        #        models=arch,
        #        optimizer=optimizer,
        #        scheduler=scheduler,
        #        epoch=epoch,
        #    )

    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    NumTrainableParams = log.logger["NumTrainableParams"]

    objective = -valLoss[-1]
    del test_dataloader, train_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "objective": objective,
        "metadata": {"TrainLoss": trainLoss, "ValLoss": valLoss},
    }
    return logger

if __name__ == '__main__':
    cfg = OmegaConf.load('conf/config_3d.yaml')
    problem = HpProblem()
    # Discrete hyperparameter (sampled with uniform prior)
    # Categorical hyperparameter (sampled with uniform prior)

    optimizers = ["Adadelta", "Adagrad", "Adam", "AdaBound", "RMSprop", "SGD"]
    schedulers = ["cosine", "step"]

    problem.add_hyperparameter((2, 16), "nr_downsamples", default_value=2)
    problem.add_hyperparameter((2, 16), "nr_residual_blocks", default_value=2)
    problem.add_hyperparameter((8, 512), "nr_latent_channels", default_value=16)
    problem.add_hyperparameter(optimizers, "optimizer", default_value="Adam")
    problem.add_hyperparameter((1e-6,1e-2,"log-uniform"), "lr", default_value=cfg.start_lr)
    problem.add_hyperparameter((0.9,1.0), "lr_scheduler_gamma", default_value=cfg.lr_scheduler_gamma)
    problem.add_hyperparameter((1,16), "batch_size", default_value=cfg.batch_size)

    print("problem: ", problem)
    sys.stdout.flush()
    
    with Evaluator.create(
        run,
        #method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            search = CBO(
                problem,
                evaluator,
                n_jobs=1,
                #verbose=1,
                initial_points=[problem.default_configuration]
            )
            results = search.search(max_evals=50)
            results.to_csv("results-1odes.csv")







