import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from lfs.utils.lfsdataloader import LFSDataLoader
from lfs.lfstrainer import LFSTrainer


train_data = dataset.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
test_data = dataset.MNIST(root = "dataset/", train = False, transform = transforms.ToTensor(), download = True)

lfsdataloader = LFSDataLoader(train_dataset=train_data, test_dataset=test_data)
train_dataloader = lfsdataloader.train_data_loader(batch_size=32, sampler="random")
test_dataloader = lfsdataloader.test_data_loader(batch_size=16, sampler="sequential")

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = Model()
lfstrainer = LFSTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=test_dataloader,
    device=torch.device("cpu"),
    optimizer_name="adam",
    scheduler_name="step_lr",
    loss_name="cross_entropy",
    num_epochs=2,
    eval_model=True,
    save_model=False,
    model_save_folder="output/",
    save_model_at_each_epoch=False,
    writer_name="tensorboard"
)

lfstrainer.train()


