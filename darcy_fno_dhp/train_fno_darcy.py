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

import hydra
from omegaconf import DictConfig
from math import ceil

import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from modulus.models.fno import FNO
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow

from validator import GridValidator

from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob

from omegaconf import OmegaConf
import numpy as np

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO

#@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def darcy_trainer(cfg: DictConfig, config: dict):
    """Training for the 2D Darcy flow benchmark problem.

    This training script demonstrates how to set up a data-driven model for a 2D Darcy flow
    using Fourier Neural Operators (FNO) and acts as a benchmark for this type of operator.
    Training data is generated in-situ via the Darcy2D data loader from Modulus. Darcy2D
    continuously generates data previously unseen by the model, i.e. the model is trained
    over a single epoch of a training set consisting of
    (cfg.training.max_pseudo_epochs*cfg.training.pseudo_epoch_sample_size) unique samples.
    Pseudo_epochs were introduced to leverage the LaunchLogger and its MLFlow integration.
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="darcy_fno")
    log.file_logging()
    initialize_mlflow(
        experiment_name=f"Darcy_FNO",
        experiment_desc=f"training an FNO model for the Darcy problem",
        run_name=f"Darcy FNO training",
        run_desc=f"training FNO for Darcy",
        user_name="Gretchen Ross",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])#lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: config["decay_rate"]**step #cfg.scheduler.decay_rate**step
    )
    norm_vars = cfg.normaliser
    normaliser = {
        "permeability": (norm_vars.permeability.mean, norm_vars.permeability.std_dev),
        "darcy": (norm_vars.darcy.mean, norm_vars.darcy.std_dev),
    }
    dataloader = Darcy2D(
        resolution=cfg.training.resolution,
        batch_size=cfg.training.batch_size,
        normaliser=normaliser,
    )
    validator = GridValidator(loss_fun=MSELoss(reduction="mean"))

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters*cfg.training.batch_size}"
        )

    # define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch+1}.")

    train_loss = np.zeros(cfg.training.max_pseudo_epochs-loaded_pseudo_epoch+1)
    valid_loss = np.zeros(int((cfg.training.max_pseudo_epochs-loaded_pseudo_epoch+1) / cfg.validation.validation_pseudo_epochs)+1)
    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        #print("pseudo_epoch: ", pseudo_epoch)
        #print("rec_results_freq: ", rec_results_freq)
        #print("validation_pseudo_epochs: ", validation_pseudo_epochs)
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            total_loss = 0.0
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                loss = forward_train(batch["permeability"], batch["darcy"])
                logger.log_minibatch({"loss": loss.detach()})
                total_loss +=loss
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            train_loss[pseudo_epoch] = total_loss / steps_per_pseudo_epoch

        # save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # validation step
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), dataloader):
                    val_loss = validator.compare(
                        batch["permeability"],
                        batch["darcy"],
                        forward_eval(batch["permeability"]),
                        pseudo_epoch,
                        logger,
                    )
                logger.log_epoch({"Validation error": total_loss / validation_iters})
                valid_loss[int(pseudo_epoch/cfg.validation.validation_pseudo_epochs)] = total_loss / validation_iters 

        # update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    history = {
        "train_loss": train_loss,
        "validation_loss": valid_loss, 
    }

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")
    return {"history": history}

def get_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        n_gpus -= 1

    is_gpu_available = n_gpus > 0

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )

    return evaluator

if __name__ == "__main__":
    # Creation of an hyperparameter problem
    problem = HpProblem()
    # Discrete hyperparameter (sampled with uniform prior)
    # Categorical hyperparameter (sampled with uniform prior)
    problem.add_hyperparameter((1e-4, 1e-2, "log-uniform"), "learning_rate", default_value=1e-3) 
    problem.add_hyperparameter((0.5, 0.95, "uniform"), "decay_rate", default_value=0.65)
    print("\nDeepHyper problem: ", problem)

    cfg = OmegaConf.load('config.yaml')

    out = darcy_trainer(cfg, RunningJob(parameters=problem.default_configuration))
    history_default = out["history"]

    print("Accuracy of the default configuration is {}".format(history_default))
    
    evaluator_1 = get_evaluator(darcy_trainer)
    search = CBO(problem, 
                evaluator_1, 
                initial_points=[problem.default_configuration]
                )
    results = search.search(max_evals=20)
    print(results)
    results.to_csv("results-learning_decay_rates_Eval20.csv")






