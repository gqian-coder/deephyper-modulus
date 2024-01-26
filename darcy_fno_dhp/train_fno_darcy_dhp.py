import mpi4py
from modulus.models.fno import FNO
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.distributed import DistributedManager

from validator import GridValidator

from deephyper.evaluator import profile, RunningJob
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.evaluator import profile, RunningJob

from Trainer import *
from omegaconf import OmegaConf
import numpy as np
import sys
import pandas as pd

import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def run(job: RunningJob):
    cfg = OmegaConf.load('config.yaml')
    param = job.parameters.copy()

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=param["decoder_layers"],
        decoder_layer_size=param["layer_size"],
        dimension=cfg.arch.fno.dimension,
        latent_channels=param["latent_channels"],
        num_fno_layers=param["fno_layers"],
        num_fno_modes=param["fno_modes"],
        padding=cfg.arch.fno.fno_modes,#param["padding"],
    ).to(dist.device)

    loss_fun  = torch.nn.MSELoss()
    optimizer = get_optimizer("Adam")#param["optimizer"])
    optimizer = optimizer(
        model.parameters(), lr=cfg.scheduler.initial_lr#param["lr"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step#param["decay_rate"]**step
    ) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(
    #    optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    #)
    print("======================== Search ===================")
    print("param: decoder_layers = {}, decoder_layer_size = {}, latent_channels = {}, num_fno_layers = {}, num_fno_modes = {}".format(param["decoder_layers"], param["layer_size"], param["latent_channels"], param["fno_layers"], param["fno_modes"]))
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
    trainer = Trainer(
        model=model,
        lossFn=torch.nn.MSELoss(),
    )
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    
    log = trainer.train(dataloader, 
                        cfg.training.max_pseudo_epochs, 
                        steps_per_pseudo_epoch, 
                        validation_iters, 
                        optimizer, 
                        scheduler, 
                        cfg.scheduler.decay_pseudo_epochs)

    trainLoss = log.logger["TrainLoss"]
    valLoss = log.logger["ValLoss"]
    NumTrainableParams = log.logger["NumTrainableParams"]

    objective = valLoss[-1]
    del dataloader, trainer 
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "objective": objective,
        "metadata": {"TrainLoss": trainLoss, "ValLoss": valLoss},
    }

if __name__ == '__main__':
    problem = HpProblem()
    # Discrete hyperparameter (sampled with uniform prior)
    # Categorical hyperparameter (sampled with uniform prior)

    activations = [
        "relu",
        "leaky_relu",
        "prelu",
        "relu6",
        "elu",
        "selu",
        "silu",
        "gelu",
        "sigmoid",
        "logsigmoid",
        "softplus",
        "softshrink",
        "softsign",
        "tanh",
        "tanhshrink",
        "threshold",
        "hardtanh",
        "identity",
        "squareplus",
    ]
    optimizers = ["Adadelta", "Adagrad", "Adam", "AdamW", "RMSprop", "SGD"]
    schedulers = ["cosine", "step"]

    #problem.add_hyperparameter((1, 16), "padding", default_value=9)
    ##problem.add_hyperparameter(activations, "lift_act", default_value="gelu")
    problem.add_hyperparameter((2, 16), "fno_layers", default_value=4)
    problem.add_hyperparameter((2, 32), "fno_modes", default_value=12)
    problem.add_hyperparameter((2, 64), "latent_channels", default_value=32)
    problem.add_hyperparameter((1, 12), "decoder_layers", default_value=1)
    problem.add_hyperparameter((16, 48), "layer_size", default_value=32)
    ##problem.add_hyperparameter(activations, "proj_act", default_value="silu")

    #problem.add_hyperparameter(optimizers, "optimizer", default_value="Adam")
    #problem.add_hyperparameter((1e-6, 1e-2), "lr", default_value=1e-3)
    ## problem.add_hyperparameter(schedulers, "scheduler", default_value='cosine')
    ##problem.add_hyperparameter((16, 256), "batch_size", default_value=64)
    ##problem.add_hyperparameter((20, 100), 'epochs', default_value=20)
    problem.add_hyperparameter((0.35, 0.95), "decay_rate", default_value=0.85)
    problem.add_hyperparameter((4, 16), "decay_epoch", default_value=8)
    
    # Baseline of training
    #results = run(RunningJob(parameters=problem.default_configuration)) 
    #print("TrainLoss: ", results["metadata"]["TrainLoss"])
    #print("ValLoss: ", results["metadata"]["ValLoss"])

    print("problem: ", problem)    
    sys.stdout.flush()
  
    results = pd.read_csv('results.csv')
    # Create a new evaluator
    with Evaluator.create(
        run,
    ) as evaluator: 
        if evaluator is not None:
            # Create a new AMBS search with strong explotation (i.e., small kappa)
            search_from_checkpoint = CBO(problem, evaluator)
            search_from_checkpoint.fit_surrogate(results)
            results_from_checkpoint = search_from_checkpoint.search(max_evals=25)

    # Initialize surrogate model of Bayesian optization
    # With results of previous search
    search_from_checkpoint.fit_surrogate(results)
    '''
    with Evaluator.create(
        run,
        #method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            search = CBO(
                problem,
                evaluator,
                #moo_scalarization_strategy="Chebyshev",
                #moo_scalarization_weight="random",
                #objective_scaler="quantile-uniform",
                #acq_func="UCB",
                #multi_point_strategy="qUCB",
                #n_jobs=1,
                #verbose=1,
                initial_points=[problem.default_configuration]
            )
            results = search.search(max_evals=200)
            results.to_csv("results-1odes-stopper.csv")
    '''  
