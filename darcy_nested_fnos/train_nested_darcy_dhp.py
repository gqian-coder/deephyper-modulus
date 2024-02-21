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
import glob
import hydra
from typing import Tuple
from omegaconf import DictConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    LaunchLogger,
    initialize_mlflow,
)

from deephyper.evaluator import profile, RunningJob
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob

from utils import NestedDarcyDataset, GridValidator
import sys
from omegaconf import OmegaConf
import pickle
import numpy as np
import gc
import torch

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

class SetUpInfrastructure:
    """Class containing most important objects

    In this class the infrastructure for training is set.

    Parameters
    ----------
    cfg : DictConfig
        config file parameters
    param:
        hyperparameter for search
    """

    def __init__(
        self, cfg: DictConfig, param, dist: DistributedManager 
    ) -> None:
        # define model, loss, optimiser, scheduler, data loader
        #model_cfg = cfg.arch[cfg.model]
        logger = PythonLogger(name="darcy_nested_fno")
        loss_fun = MSELoss(reduction="mean")
        norm = {
            "permeability": (
                cfg.normaliser.permeability.mean,
                cfg.normaliser.permeability.std,
            ),
            "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std),
        }
        self.training_set = NestedDarcyDataset(
            mode="train",
            data_path=cfg.training.training_set,
            norm=norm,
            log=logger,
        )
        self.valid_set = NestedDarcyDataset(
            mode="train",
            data_path=cfg.validation.validation_set,
            norm=norm,
            log=logger,
        )

        print(
            "Training set contains {len(self.training_set)} samples, validation set contains {len(self.valid_set)} samples."
        )

        train_sampler = DistributedSampler(
            self.training_set,
            num_replicas=dist.world_size,
            rank=dist.local_rank,
            shuffle=True,
            drop_last=False,
        )

        valid_sampler = DistributedSampler(
            self.valid_set,
            num_replicas=dist.world_size,
            rank=dist.local_rank,
            shuffle=True,
            drop_last=False,
        )

        self.train_loader = DataLoader(
            self.training_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            sampler=train_sampler,
        )
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            sampler=valid_sampler,
        )
        self.validator = GridValidator(loss_fun=loss_fun, norm=norm)

        self.model = FNO(
            in_channels=cfg.arch.fno.in_channels,
            out_channels=cfg.arch.decoder.out_features,
            decoder_layers=param["decoder_layers"],
            decoder_layer_size=param["layer_size"],
            dimension=cfg.arch.fno.dimension,
            latent_channels=param["latent_channels"],
            num_fno_layers=param["fno_layers"],
            num_fno_modes=param["fno_modes"],
            padding=cfg.arch.fno.fno_modes,
        ).to(dist.device)

        self.optimizer = Adam(self.model.parameters(), lr=cfg.scheduler.initial_lr)
        self.scheduler = lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
        )
        print("Finish FNO")  
        print("self.model: ", self.model)

        # define forward for training and inference
        @StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            use_amp=False,
            use_graphs=False,
        )
        def _forward_train(invars, target):
            pred = self.model(invars)
            loss = loss_fun(pred, target)
            return loss

        @StaticCaptureEvaluateNoGrad(
            model=self.model, use_amp=False, use_graphs=False
        )
        def _forward_eval(invars):
            return self.model(invars)

        self.forward_train = _forward_train
        self.forward_eval = _forward_eval


def TrainModel(cfg: DictConfig, base: SetUpInfrastructure) -> None:
    """Training Loop

    Parameters
    ----------
    cfg : DictConfig
        config file parameters
    base : SetUpInfrastructure
        important objects
    """
    logger = Logger("./log/")
    logger.log(
            "NumTrainableParams",
            sum(p.numel() for p in base.model.parameters() if p.requires_grad),
    )
    min_valid_loss = 9.0e9
    print("number of training steps: ", len(base.train_loader))
    print("validation iters: ", len(base.valid_loader))
    for epoch in range(1, cfg.training.max_epochs + 1):
        # Wrap epoch in launch logger for console / MLFlow logs
        runningLoss = 0.0
        for batch in base.train_loader:
            loss = base.forward_train(batch["permeability"], batch["darcy"])
            runningLoss += loss.detach()
        runningLoss = runningLoss / len(base.train_loader) 
        print("epoch", epoch, ", runningLoss: ", runningLoss.cpu().numpy())
        sys.stdout.flush()

        # validation
        if (
            epoch % cfg.validation.validation_epochs == 0
            or epoch % cfg.training.rec_results_freq == 0
            or epoch == cfg.training.max_epochs
        ):
            total_loss = 0.0
            for batch in base.valid_loader:
                loss = base.validator.compare(
                    batch["permeability"],
                    batch["darcy"],
                    base.forward_eval(batch["permeability"]),
                    epoch,
                    logger,
                )
                total_loss += loss * batch["darcy"].shape[0] / len(base.valid_set)
            
            valLoss = total_loss / len(base.valid_loader) 
            logger.log("Epoch", epoch)
            logger.log("TrainLoss", runningLoss.cpu().numpy())
            logger.log("ValLoss", valLoss.cpu().numpy())
            logger.save('epoch'+str(epoch))
            #log.log_epoch({"Validation error": total_loss})

        # update learning rate
        if epoch % cfg.scheduler.decay_epochs == 0:
            base.scheduler.step()

    return logger


def run(job: RunningJob):
    cfg = OmegaConf.load('config_dhp.yaml')
    param = job.parameters.copy()
   # print(cfg)
    print("FNO model inputs: in_channels={}, out_channels={}, decoder_layers={}, decoder_layer_size={}, dimension={}, latent_channels={}, num_fno_layers={}, num_fno_modes={}, padding={}".format(cfg.arch.fno.in_channels, cfg.arch.decoder.out_features, param["decoder_layers"], param["layer_size"], cfg.arch.fno.dimension, param["latent_channels"], param["fno_layers"], param["fno_modes"], cfg.arch.fno.fno_modes))

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    """Training for the 2D nested Darcy flow problem.

    This training script demonstrates how to set up a data-driven model for a nested 2D Darcy flow
    using nested Fourier Neural Operators (nFNO, https://arxiv.org/abs/2210.17051). nFNOs are
    basically a concatenation of individual FNO models. Individual FNOs can be trained independently
    and in any order. The order only gets important for fine tuning (tba) and inference.
    """

    # set up infrastructure
    base = SetUpInfrastructure(cfg, param, dist)

    # catch restart in case checkpoint exists
    #loaded_epoch = load_checkpoint(**base.ckpt_args, device=dist.device)

    # train model
    log = TrainModel(cfg, base)

    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    NumTrainableParams = log.logger["NumTrainableParams"]

    del base
    gc.collect()
    torch.cuda.empty_cache()

    objective = -valLoss[-1]
    return {
        "objective": objective,
        "metadata": {"TrainLoss": trainLoss, "ValLoss": valLoss},
    }


if __name__ == "__main__":
    problem = HpProblem() 
    problem.add_hyperparameter((2, 16), "fno_layers", default_value=4)
    problem.add_hyperparameter((2, 32), "fno_modes", default_value=12)
    problem.add_hyperparameter((8, 64), "latent_channels", default_value=32)
    problem.add_hyperparameter((1, 12), "decoder_layers", default_value=1)
    problem.add_hyperparameter((16, 64), "layer_size", default_value=32)

    print("problem: ", problem)
    sys.stdout.flush()

    '''
    results = pd.read_csv('results.csv')
    # Create a new evaluator
    with Evaluator.create(
        run,
    ) as evaluator:
        if evaluator is not None:
            # Create a new AMBS search with strong explotation (i.e., small kappa)
            search_from_checkpoint = CBO(problem, evaluator)
            search_from_checkpoint.fit_surrogate(results)
            results_from_checkpoint = search_from_checkpoint.search(max_evals=50)

    # Initialize surrogate model of Bayesian optization
    # With results of previous search
    '''
    with Evaluator.create(
        run,
    ) as evaluator:
        if evaluator is not None:
            search = CBO(
                problem,
                evaluator,
                initial_points=[problem.default_configuration]
            )
            results = search.search(max_evals=200)
            results.to_csv("results-1odes-stopper.csv")
    
