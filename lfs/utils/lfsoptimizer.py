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

        self.optimizer_name = optimizer_name
        self.model_params = model_params
        self.lr = config.LEARNING_RATE
        self.betas = config.BETAS
        self.eps = config.EPS
        self.weight_decay = config.WEIGHT_DECAY

    def get_req_optimizer(self):

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

        