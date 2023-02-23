import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F

class LFSLosses:
    
    '''
    A class which contains a list of losses that checks the loss_name and then returns 
    the loss function required accordingly from the get_req_loss_func function.

    Input:
    loss_name : str : Name of the loss function required. Input comes from LFSTrainerUtils class fetch_lossfunc() function. Eg: cross_entropy, bce, mse, etc...
    '''

    def __init__(
        self,
        loss_name
        ):
        self.loss_name=loss_name

    def get_req_lossfunc(self):
        
        '''
        A function that checks the loss_name and returns the respective loss function object
        
        Input:
        loss_name : str : Name of the loss function required. Input from self

        Returns:
        Object of the loss function required
        '''

        # Checking the loss_name if it is cross_entropy or bce or mse3 or etc... and returning the loss function specified
        if self.loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        
        elif self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()

        elif self.loss_name == "mse":
            return nn.MSELoss()