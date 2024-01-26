import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from modulus.models.fno import FNO
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.datapipes.benchmarks.darcy import Darcy2D
from validator import GridValidator
from math import ceil
import sys

def get_optimizer(name):
    if name == "Adadelta":
        optimizer = torch.optim.Adadelta
    elif name == "Adagrad":
        optimizer = torch.optim.Adagrad
    elif name == "Adam":
        optimizer = torch.optim.Adam
    elif name == "AdamW":
        optimizer = torch.optim.AdamW
    elif name == "RMSprop":
        optimizer = torch.optim.RMSprop
    elif name == "SGD":
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {name} does not exist.")
    return optimizer


def get_scheduler(name, optimizer):
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50)
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    else:
        raise ValueError(f"Scheduler {name} does not exist.")
    return scheduler


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

class Trainer:
    def __init__(self, model, lossFn):
        self.model = model
        self.lossFn = lossFn

    def train(
        self,
        dataloader,
        epochs,
        steps_per_pseudo_epoch,
        validation_iters, 
        optimizer,
        scheduler,
        scheduler_epochs,
    ):
        #optimizer = get_optimizer(optimizer)
        #optimizer = optimizer(
        #    self.model.parameters(), lr=learningRate, weight_decay=weight_decay
        #)
        #scheduler = get_scheduler(scheduler, optimizer)

        # define forward passes for training and inference
        @StaticCaptureTraining(
            model=self.model, optim=optimizer, use_amp=False, use_graphs=False
        )
        def forward_train(invars, target):
            pred = self.model(invars)
            loss = self.lossFn(pred, target)
            return loss
        @StaticCaptureEvaluateNoGrad(
            model=self.model, use_amp=False, use_graphs=False
        )
        def forward_eval(invars):
            return self.model(invars)

        logger = Logger("./log/")
        logger.log(
            "NumTrainableParams",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        print("NumTrainableParams: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("steps_per_pseudo_epoch: ", steps_per_pseudo_epoch)
        validator = GridValidator(loss_fun=torch.nn.MSELoss(reduction="mean"))
        for pseudo_epoch in range(epochs + 1):
            print("epoch = {}/{}".format(pseudo_epoch, epochs))
            # Wrap epoch in launch logger for console / MLFlow logs
            runningLoss = 0.0
            for iter_s, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                if (iter_s==0 and pseudo_epoch==0):
                    print(batch["permeability"].data.cpu().numpy().shape, batch["darcy"].data.cpu().numpy().shape)
                loss = forward_train(batch["permeability"], batch["darcy"])
                runningLoss += loss.detach()
            
            runningLoss = runningLoss / steps_per_pseudo_epoch
            print("runningLoss: ", runningLoss.cpu().numpy())
            sys.stdout.flush()

            ## save checkpoint
            #if pseudo_epoch % cfg.training.rec_results_freq == 0:
            #    save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

            # validation step
            if pseudo_epoch % 2 == 0:
                valLoss = 0.0 
                for _, batch in zip(range(validation_iters), dataloader):
                    val_loss = validator.compare(
                        batch["permeability"],
                        batch["darcy"],
                        forward_eval(batch["permeability"]),
                        pseudo_epoch,
                        logger,
                    )
                    valLoss += val_loss.detach()
                valLoss = valLoss / validation_iters

            # update learning rate
            if pseudo_epoch % scheduler_epochs == 0:
                scheduler.step()

            logger.log("Epoch", pseudo_epoch)
            logger.log("TrainLoss", runningLoss.cpu().numpy())
            logger.log("ValLoss", valLoss.cpu().numpy())

        logger.save('epoch'+str(epochs))
        return logger

if __name__ == "__main__":
    cfg = OmegaConf.load('config.yaml')

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

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
    loss_fun = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
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
    trainer = Trainer(
        model=model,
        lossFn=loss_fun,
    )
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    
    log = trainer.train(dataloader, 80, steps_per_pseudo_epoch, validation_iters, optimizer, scheduler, cfg.scheduler.decay_pseudo_epochs)
    print("TrainLoss: ", log.logger["TrainLoss"])
    print("ValLoss: ", log.logger["ValLoss"])
