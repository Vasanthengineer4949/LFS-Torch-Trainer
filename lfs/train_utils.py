import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from lfs.utils import lfsoptimizer, lfsscheduler, lfslosses
from torch.utils.tensorboard import SummaryWriter
import onnx
import onnxruntime as ort
import numpy as np

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
    
    def model_save(
        self,
        file_path : str =None, 
        model : nn.Module =None, 
        optimizer : torch.optim =None,
        scheduler : torch.optim.lr_scheduler = None,
        epoch : int =None, 
        train_loss : list =None, 
        val_loss : list =None):

        '''
        A function to save the model, optimizer, scheduler states along with the epoch and losses
        
        Input:
        file_path : str : The path in which the saving process should take place
        model : nn.Module : Model object need to be stored
        optimizer : torch.optim : Optimizer object that is needed to be stored
        scheduler : torch.optim.lr_scheduler : Scheduler object need to be stored
        epoch : int : Epoch number
        train_loss : list : List of training losses
        val_loss : list : List of validation losses
        
        Returns:
        None
        
        Stores:
        States of Model, Optimizer, Scheduler, Epoch Number, Training Loss, Validation Loss
        '''

        torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": val_loss
                    },  
                    file_path
                )

    def tensorboard_write(
        self, 
        train_loss_epoch : list = None,
        val_loss_epoch : list = None,
        evaluate_model : bool = False
        ):

        '''
        A function to create a tensorboard and write the losses in it
        
        Input:
        train_loss_epoch : list : List of training losses by epochs
        val_loss_epoch : list : List of validation losses by epochs
        evaluate_model : bool : Boolean value to indicate if the model is enabled with validation
        
        Writes:
        Training and Validation losses in tensorboard
        '''

        # Initializing the SummaryWriter class
        writer = SummaryWriter()

            # Traversing through the loss in the train_epoch_loss list and if specified True for eval_model then val_epoch_loss as well
        for loss_write in range(len(train_loss_epoch)):

                # Adding the train_loss as a scalar value along with the epoch number to generate the train_epoch_loss graph
                writer.add_scalar("Loss/Train", train_loss_epoch[loss_write], loss_write)

                # Checking if there is model validation been performed
                if evaluate_model==True:

                    # Adding the val_loss as a scalar value along with the epoch number to generate the val_epoch_loss graph
                    writer.add_scalar("Loss/Val", val_loss_epoch[loss_write], loss_write)

    def model_save_onnx(
        self,
        file_path : str = None,
        model : nn.Module = None,
        dummy_input : torch.Tensor = None,
        input_names : list = None,
        output_names : list = None,
        verify_onnx_model : bool = False
        ):

        '''
        A function to export model as onnx file and also it checks if the model is well formed
        
        Input:
        file_path : str : The path in which the onnx file is to be stored
        model : nn.Module: The model which needs to be stored as onnx
        dummy_input : torch.Tensor : A dummy tensor which shows the shape of the input
        input_names : list : The names of the input key
        output_names : list : The names output key
        verify_onnx_model : bool : To check if the onnx model working needs to be verified
        '''
        torch.onnx.export(
                model,
                dummy_input,
                f = file_path,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
        )
        onnx_model = onnx.load(file_path)
        onnx.checker.check_model(onnx_model)





