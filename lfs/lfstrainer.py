import lfs.config as config
import lfs.train_utils as train_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

class LFSTrainer(nn.Module):
    
    def __init__(
        self,
        model : nn.Module = None,
        train_dataloader : DataLoader = None, 
        test_dataloader : DataLoader = None, 
        val_dataloader : DataLoader = None,
        device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optimizer_name : str = config.OPTIMIZER_NAME, 
        scheduler_name : str = config.SCHEDULER_NAME,
        loss_name : str = config.LOSS_NAME,
        writer_name : str = "tensorboard",
        num_epochs : int = config.NUM_EPOCHS,
        eval_model : bool = False,
        save_model : bool = False,
        model_save_folder : str = None,
        save_model_at_each_epoch : bool = False,
        save_as_onnx : bool =False,
        *args, 
        **kwargs):

        '''
        A wrapper class for torch training loop and the supplements required in the loop
        
        Input:
        model: nn.Module: Class of the model to be trained. Default: None
        train_dataloader: DataLoader : Training DataLoader. Default : None
        test_dataloader: DataLoader : Testing DataLoader. Default: None
        val_dataloader: DataLoader : Validation DataLoader. Default: None
        device: torch.device(): Device to be used by torch. Default: It will find if cuda or cpu
        optimizer_name: str : Name of the optimizer required. Default: "adam"
        scheduler_name: str : Name of the scheduler required. Default: "steplr"
        loss_name : str : Name of the loss function required. Default: "cross_entropy"
        writer_name : str : Name of the writer to be used. Default: "tensorboard"
        num_epochs : int : Number of epochs to be trained for. Default: 5
        eval_model : bool : To check if the model needs to be validated. Default: False
        save_model : bool : To check if the model needs to be saved. Default: False
        model_save_folder: str : Path to store the model to be saved. Default: None
        save_model_at_each_epoch : bool : To check if the model needs to be saved at each end of epoch. Default: False  
        save_as_onnx : bool : To check if the model needs to be saved as onnx file
        verify_onnx_model : bool To check if the saved onnx_model needs to be verified
        '''

        super().__init__()

        self.model = model

        self.train_dataloader = train_dataloader 
        self.test_dataloader = test_dataloader 
        self.val_dataloader = val_dataloader  

        self.device = device 

        self.lfs_train_utils = train_utils.LFSTrainerUtils()

        self.optimizer = self.lfs_train_utils.fetch_optimizer(
                                            optimizer_name, 
                                            self.model.parameters()
                                            )

        self.scheduler = self.lfs_train_utils.fetch_scheduler(
                                            scheduler_name, 
                                            self.optimizer
                                            )

        self.loss = self.lfs_train_utils.fetch_loss(loss_name)
        # self.writer = self.lfs_train_utils.create_writer(writer_name)
        self.num_epochs = num_epochs
        self.eval_model = eval_model
        self.save_model = save_model
        self.save_model_at_each_epoch = save_model_at_each_epoch
        self.model_save_folder = model_save_folder
        self.writer_name = writer_name
        self.inp_shape = []
        self.save_as_onnx = save_as_onnx
        self.args = args
        self.kwargs = kwargs

    def train_one_step(self, inp, labels):

        '''
        A function to perform the model training for one step.
        
        Inputs:
        inp : Input data. Can be of any type. But needs to be handled accordingly in the model forward function.
        labels: target labels (i.e) the output data.

        Returns
        loss: Loss at the step
        '''

        # Setting the gradients of the optimizer to None
        self.optimizer.zero_grad()

        # Converting the input data and labels to the device of the torch
        inp = inp.to(self.device)
        labels = labels.to(self.device)

        try:

            # Getting the outputs from the model by passing the inputs to the model
            pred = self.model(inp)

            # Calculate the loss by passing the prediceted labels and actual labels to the loss function
            loss = self.loss(pred, labels)

        except Exception as e:
            raise ValueError("Only one output value should be acquired from model")
    
        # Folowing two steps are for backpropagation
        loss.backward() # To compute the gradient of loss 
        self.optimizer.step() # To update the parameter at this step. Performs a single optimization step
        return loss

    def train_one_epoch(self):

        '''
        A function which is used to train the model for one epoch
        
        Input:
        train_dataloader: Training DataLoader object from self
        
        Returns:
        epoch_loss: Loss at each epoch
        '''

        self.model.to(self.device)

        # Setting the model in training mode
        self.model.train()

        #Setting the epoch_loss at 0 in which at each epoch the loss will be added to
        epoch_loss = 0

        # Traversing through the training dataloader
        for idx, (inp, targets) in enumerate(tqdm(self.train_dataloader)):

            # Performing the function of each step by calling the train_one_step function
            loss = self.train_one_step(inp, targets)

            # Adding the loss from the step to the epoch_loss
            epoch_loss += loss
        
        # Stepping the scheduler to update the learning_rate at the end of the each epoch
        self.scheduler.step()

        # Returning the loss at each epoch by dividing the loss by the length of the train_dataloader so we can get the loss at each epoch
        return epoch_loss/len(self.train_dataloader)
    
    def val_one_step(self, inp, labels):

        '''
        Performs validation process at one step
        
        Inputs:
        inp : Inp data same as training input data format
        labels: Target outputs for the model to predict to
        
        Returns:
        loss: The validation loss at each step
        '''

        # Converting the inpinputut data and labels to the torch device specified
        inp = inp.to(self.device)
        labels = labels.to(self.device)

        # Getting a dummy input tensor for onnx 
        if self.inp_shape == []:
            self.inp_shape = list(inp.shape)[1:]
            self.dummy_tensor = torch.randn(self.inp_shape, device=self.device)

        try:
            # Getting the outputs from the model by passing the inputs to the model
            pred = self.model(inp)
            # Calculating the loss by passing the predicted labels and actual labels as input to the loss function
            loss = self.loss(pred, labels)
        except Exception as e:
            raise ValueError("Only one output value should be acquired from model")
        
        # Return the validation loss at the ste[]
        return loss

    def val_one_epoch(self):

        # Setting torch to no_grad mode to avoid gradient computations since this is not needed
        with torch.no_grad():

            # Setting the model in eval mode
            self.model.eval()

            # val_epoch_loss variable to store the losses at each step and epoch
            val_epoch_loss = 0

            # Checking if the val_dataloader is not None to avoid error
            if self.val_dataloader != None:

                # Traversing through val_dataloader data
                for idx, (inp, targets) in enumerate(tqdm(self.val_dataloader)):

                    # Performing validation functions for number of steps by calling the val_one_step
                    val_loss = self.val_one_step(inp, targets)

                    # Aggregating the loss of the step with the val_epoch_loss
                    val_epoch_loss += val_loss

                # Returning the val_epoch_loss after dividing it byt the length of the validation dataloader to get the validation loss at each epoch
                return val_epoch_loss/len(self.val_dataloader)
            else:
                raise ValueError("Val Data Loader not provided")

    def train(self):

        '''
        The wrapper function for the whole training loop
        
        Inputs:
        No user inputs
        Functions from the class and params in init function are inputs

        Returns:
        None
        
        Stores:
        model, optimizer, scheduler state_dict along with trianing and validation loss if specified
        '''

        # Lists to store the train and validation loss at each epoch to enable monitoring
        train_loss_epoch = []
        val_loss_epoch = []

        # Training process starts and runs till the number of epochs specified
        for epoch in range(self.num_epochs):

            # Getting the training loss at each epoch and appending it to the train_epoch_loss list
            train_loss = self.train_one_epoch()
            train_loss_epoch.append(train_loss.item())

            # Checking if the eval_model param is set to True and if it is set to True then model validation is performed
            if self.eval_model == True:
                
                # Getting the validation loss at the end of each epoch and appending it to the val_epoch_loss list
                val_loss = self.val_one_epoch()
                val_loss_epoch.append(val_loss.item())

                # Printing the training and validation loss at each epoch
                print("Train Loss at {} epoch is: {}\n Validation Loss at {} epoch is: {}\n======================================================================================================\n".format(epoch, train_loss.item(), epoch, val_loss.item()))
            else:
                # Printing the training loss at the end of each epoch
                print("Train Loss at {} epoch is: {}\n======================================================================================================\n".format(epoch, train_loss.item()))

            # Checking if the model needs to be saved
            if self.save_model==True:

                # Checking if the model needs to be saved at each epoch
                if self.save_model_at_each_epoch == True:

                    # Checking if the model_save_folder is not None to ensure there is a path to the folder to save to
                    if self.model_save_folder != None:

                        # Saving the state of model, optimizer, scheduler along with the epoch and the training and validation loss with file_path as:
                        # file_path = folder_path/epoch_[epoch_number]_checkpoint.pth 
                        self.lfs_train_utils.model_save(
                        file_path=self.model_save_folder+f"epoch_{epoch}_checkpoint.pth",
                        model = self.model,
                        optimizer = self.optimizer,
                        scheduler = self.scheduler,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss
                    )

                    else:
                        raise ValueError("Model Saving Folder is not provided")
                else:
                    pass
            else:
                pass
        
        # To check if the model needs to be saved
        if self.save_model==True:

            # To check if the model need not be saved at each epoch end
            if self.save_model_at_each_epoch==False:

                # To ensure that there is a foler path for saving
                if self.model_save_folder != None:

                    # Saving the states of model, optimizer, scheduler along with the epoch, training and validation loss with file_path as:
                    # file_path = folder_path/final_checkpoint.pth
                    self.lfs_train_utils.model_save(
                        file_path=self.model_save_folder+"final_checkpoint.pth",
                        model = self.model,
                        optimizer = self.optimizer,
                        scheduler = self.scheduler,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss
                    )
                else:
                    raise ValueError("Model Saving Folder is not provided")
            else:
                pass
        else:
            pass
        
        # Checking if the writer/logger is tensorboard
        if self.writer_name=="tensorboard":

            self.lfs_train_utils.tensorboard_write(
                train_loss_epoch=train_loss_epoch,
                val_loss_epoch=val_loss_epoch,
                evaluate_model=self.eval_model
            )

    
        # To check if the model need be saved as onnx
        if self.save_as_onnx == True:
            self.lfs_train_utils.model_save_onnx(
                file_path=self.model_save_folder+"model.onnx",
                model=self.model,
                dummy_input=self.dummy_tensor,
                input_names=["img_tensors"],
                output_names=["labels"]
            )
