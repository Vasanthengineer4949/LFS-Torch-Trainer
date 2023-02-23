import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as schedulers

class LFSScheduler:

    '''
    A class which contains a list of schedulers that checks the scheduler_name and then returns 
    the optimizer required accordingly from the get_req_scheduler function.

    Input: Input comes from LFSTrainerUtils class fetch_scheduler() function.
    scheduler_name : str : Name of the scheduler required. Eg: steplr, lr_on_plateau, etc...
    optimizer : torch.optim.optimizer_using : Object of the optimizer being used
    '''

    def __init__(
        self,
        scheduler_name:str=config.SCHEDULER_NAME,
        optimizer:torch.optim=None
        ):
        self.scheduler_name = scheduler_name
        self.optimizer = optimizer
        self.step_size = 30
        self.gamma = 0.1
        self.last_epoch = config.LAST_EPOCH
        self.mode = config.MODE
        self.factor = config.FACTOR
        self.patience = config.PATIENCE
        self.threshold = config.THRESHOLD
        self.cooldown = config.COOLDOWN
        self.eps = config.EPS

    def get_req_scheduler(self):
        
        '''
        A function that checks the scheduler_name and returns the respective scheduler object
        
        Input:Input from self
        scheduler_name : str : Name of the scheduler required.
        optimizer : torch.optim.optimizer_using : Object of the optimizer being used
        other_params: Other params depend on the scheduler being used 

        Returns:
        Object of the scheduler required
        '''

        if self.scheduler_name=="step_lr":
            return schedulers.StepLR(
                optimizer=self.optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
                last_epoch=self.last_epoch,
                verbose=False
            )
        
        elif self.scheduler_name=="lr_on_plateau":
            return schedulers.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode=self.mode,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                cooldown=self.cooldown,
                eps=self.eps,
                verbose=False
            )

        else:
            return schedulers.StepLR(
                optimizer=self.optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
                last_epoch=self.last_epoch,
                verbose=False
            )