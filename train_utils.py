import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from lfs.utils import lfsoptimizer, lfsscheduler, lfslosses

class LFSTrainerUtils(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fetch_optimizer(
        self, 
        optimizer_name:str = config.OPTIMIZER_NAME, 
        model_params:nn.Module.parameters=None
        ):
        optimizers = lfsoptimizer.LFSOptimizers(optimizer_name, model_params)
        optimizer = optimizers.get_req_optimizer()
        return optimizer

    def fetch_scheduler(
        self, 
        scheduler_name:str = config.SCHEDULER_NAME, 
        optimizer:torch.optim=None
        ):
        schedulers = lfsscheduler.LFSScheduler(scheduler_name, optimizer)
        scheduler = schedulers.get_req_scheduler()
        return scheduler

    def fetch_loss(
        self, 
        loss_name:str = "cross_entropy"
        ):
        loss_funcs = lfslosses.LFSLosses(loss_name)
        loss_func = loss_funcs.get_req_lossfunc()
        return loss_func
    
    # def create_writer(self, writer_name:str = "tensorboard"):
    #     writer = lfswriter.LFSWriter(writer_name)
    #     return writer()
    
