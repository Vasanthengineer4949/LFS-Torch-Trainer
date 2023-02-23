import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from lfs.utils import lfsoptimizer, lfsscheduler, lfslosses

class LFSTrainerUtils(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        '''
        A Trainer utils module from which all supplements such as optimizer, scheduler, loss,
        etc... is provided to the Trainer
        '''
        
        super().__init__(*args, **kwargs)

    def fetch_optimizer(
        self, 
        optimizer_name:str = config.OPTIMIZER_NAME, 
        model_params:nn.Module.parameters=None
        ):

        '''
        A function which is used to fetch the optimizer required.
        
        Input:
        optimizer_name: str: The name of the optimizer required. Eg: adam, adamw Default: adam
        model_params: nn.Module.parameters() object: Parameters of the model

        Returns:
        optimizer - The optimizer expected as an optimizer object        
        '''

        # Initializing the LFSOptimizer class from the lfsoptimizer.py which has the different optimizers
        optimizers = lfsoptimizer.LFSOptimizers(optimizer_name, model_params)
        # get_req_optimizer checks the optimizer_name and returns the optimizer accordingly
        optimizer = optimizers.get_req_optimizer()
        return optimizer

    def fetch_scheduler(
        self, 
        scheduler_name:str = config.SCHEDULER_NAME, 
        optimizer:torch.optim=None
        ):

        '''
        A function which is used to fetch the scheduler required.
        
        Input:
        scheduler_name: str: The name of the scheduler required. Eg: step_lr, lr_on_plateau Default: step_lr
        optmizer : torch.optim.optimizer_using object: Object of the optimizer being used 

        Returns:
        scheduler - The scheduler expected as a scheduler object        
        '''

        # Initializing the LFSScheduler class from the lfsscheduler.py which has the different schedulers
        schedulers = lfsscheduler.LFSScheduler(scheduler_name, optimizer)
        # get_req_scheduler checks the scheduler_name and returns the scheduler accordingly
        scheduler = schedulers.get_req_scheduler()
        return scheduler

    def fetch_loss(
        self, 
        loss_name:str = config.LOSS_NAME
        ):

        '''
        A function which is used to fetch the loss function required.
        
        Input:
        loss_name: str: The name of the loss_function required. Eg: cross_entropy, bce, mse Default: cross_entropy

        Returns:
        loss_func - The loss function expected as a loss function object        
        '''

        # Initializing the LFSLosses class from the lfslosses.py which has the different loss functions
        loss_funcs = lfslosses.LFSLosses(loss_name)
        # get_req_lossfunc checks the loss_name and returns the loss function accordingly
        loss_func = loss_funcs.get_req_lossfunc()
        return loss_func
    
