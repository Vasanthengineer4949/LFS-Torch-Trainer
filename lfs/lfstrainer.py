# Input comes as torch datasets for train and validation also add test if present
import lfs.config as config
import lfs.train_utils as train_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from lfs.utils.lfsdataloader import LFSDataLoader

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
        *args, 
        **kwargs):

        super().__init__()

        self.model = model

        self.train_dataloader = train_dataloader # done
        self.test_dataloader = test_dataloader # done
        self.val_dataloader = val_dataloader  # done

        self.device = device # done

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
        self.args = args
        self.kwargs = kwargs

    def train_one_step(self, inp, labels):
        self.optimizer.zero_grad()
        inp = inp.to(self.device)
        labels = labels.to(self.device)
        try:
            pred = self.model(inp)
            loss = self.loss(pred, labels)
        except Exception as e:
            raise ValueError("Only one output value should be acquired from model")
        loss.backward()
        self.optimizer.step()
        return loss

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        for idx, (inp, targets) in enumerate(self.train_dataloader):
            loss = self.train_one_step(inp, targets)
            epoch_loss += loss
        self.scheduler.step()
        return epoch_loss/len(self.train_dataloader)
    
    def val_one_step(self, inp, labels):
        inp = inp.to(self.device)
        labels = labels.to(self.device)
        try:
            pred = self.model(inp)
            loss = self.loss(pred, labels)
        except Exception as e:
            raise ValueError("Only one output value should be acquired from model")
        return loss

    def val_one_epoch(self):
        with torch.no_grad():
            self.model.eval()
            val_epoch_loss = 0
            if self.val_dataloader != None:
                for idx, (inp, targets) in enumerate(self.val_dataloader):
                    val_loss = self.val_one_step(inp, targets)
                    val_epoch_loss += val_loss
                return val_epoch_loss/len(self.val_dataloader)
            else:
                raise ValueError("Val Data Loader not provided")

    def train(self):
        train_loss_epoch = []
        val_loss_epoch = []
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            train_loss_epoch.append(train_loss.item())
            if self.eval_model == True:
                val_loss = self.val_one_epoch()
                val_loss_epoch.append(val_loss.item())
                print("Train Loss at {} epoch is: {}\n Validation Loss at {} epoch is: {}\n======================================================================================================\n".format(epoch, train_loss.item(), epoch, val_loss.item()))
            else:
                print("Train Loss at {} epoch is: {}\n======================================================================================================\n".format(epoch, train_loss.item()))

        if self.writer_name=="tensorboard":
            writer = SummaryWriter()
            for loss_write in range(len(train_loss_epoch)):
                writer.add_scalar("Loss/Train", train_loss_epoch[loss_write], loss_write)
                if self.eval_model==True:
                    writer.add_scalar("Loss/Val", val_loss_epoch[loss_write], loss_write)
            train_loss_epoch.clear()
            val_loss_epoch.clear()

            if self.save_model==True:
                if self.save_model_at_each_epoch == True:
                    if self.model_save_folder != None:
                        torch.save(
                            {
                                "model": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "lr_schfrom torch.utils.tensorboard import SummayWritereduler": self.scheduler.state_dict(),
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "validation_loss": val_loss
                                },
                        self.model_save_folder + f"epoch_{epoch}_checkpoint.pth"
                            )
                    else:
                        raise ValueError("Model Saving Folder is not provided")
                else:
                    pass
            else:
                pass
        
        if self.save_model==True:
            if self.save_model_at_each_epoch==False:
                if self.model_save_folder != None:
                    torch.save(
                            {
                                "model": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "lr_scheduler": self.scheduler.state_dict(),
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "validation_loss": val_loss
                                },
                        self.model_save_folder + f"final_checkpoint.pth"
                    )
                else:
                    raise ValueError("Model Saving Folder is not provided")
            else:
                pass
        else:
            pass


                        
                        
        
    
            

    

