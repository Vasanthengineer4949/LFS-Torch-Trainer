import lfs.config as config
import torch
import torch.optim as optimizers
import torch.nn as nn
import torch.nn.functional as F

class LFSOptimizers:

    def __init__(
        self, 
        optimizer_name:str=config.OPTIMIZER_NAME,
        model_params:nn.Module.parameters=None
        ):

        '''
        A class which contains a list of optimizers that checks the optimizer_name and then returns 
        the optimizer required accordingly from the get_req_optimizer function.

        Input: Input comes from LFSTrainerUtils class fetch_optimizer() function.
        optimizer_name : str : Name of the optimizer required. Eg: adam, adamw, etc...
        model_params : nn.Module.parameters() : Model parameters to update
        '''
        
        self.optimizer_name = optimizer_name
        self.model_params = model_params
        self.lr = config.LEARNING_RATE
        self.betas = config.BETAS
        self.eps = config.EPS
        self.weight_decay = config.WEIGHT_DECAY

    def get_req_optimizer(self):

        '''
        A function that checks the optimizer_name and returns the respective optimizer object
        
        Input:Input from self
        optimizer_name : str : Name of the optimizer required.
        model_params : nn.Module.parameters() : Model parameters to update
        lr : float : Learning Rate at which the optimizer should start
        other_params: Other params depend on the optimizer being used 

        Returns:
        Object of the optimizer required
        '''

        # Checking the optimizer_name if it is adam or adamw or etc... and returning the optimizer specified
        if self.optimizer_name == "adam":
            return optimizers.Adam(
                params=self.model_params, 
                lr=self.lr, 
                betas=self.betas, 
                eps=self.eps,
                weight_decay=self.weight_decay)
        
        elif self.optimizer_name == "adamw":
            return optimizers.AdamW(
                params=self.model_params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay
            )
        
        else:
            return optimizers.Adam(
                params=self.model_params, 
                lr=self.lr, 
                betas=self.betas, 
                eps=self.eps,
                weight_decay=self.weight_decay)

        