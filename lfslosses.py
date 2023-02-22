import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F

class LFSLosses:

    def __init__(
        self,
        loss_name
        ):
        self.loss_name=loss_name

    def get_req_lossfunc(self):
        
        if self.loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        
        elif self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()

        elif self.loss_name == "mse":
            return nn.MSELoss()