import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler

class LFSDataLoader():

    def __init__(self, train_dataset, val_dataset=None, test_dataset=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset

    def train_data_loader(self, batch_size=config.TRAIN_BATCH_SIZE, sampler=config.TRAIN_SAMPLER):

        if sampler==config.TRAIN_SAMPLER:
            train_sampler = RandomSampler(self.train_data)

        else:
            raise NotImplementedError("Try a different sampler")

        train_dataloader =  DataLoader(
            dataset=self.train_data,
            sampler=train_sampler,
            batch_size = batch_size,
            num_workers=config.TRAIN_DATALOADER_NUM_WORKERS
        )

        return train_dataloader

    def val_data_loader(self, batch_size=config.TEST_VAL_BATCH_SIZE, sampler=config.TEST_VAL_SAMPLER):

        if sampler==config.TEST_VAL_SAMPLER:
            val_sampler = SequentialSampler(self.val_data)

        else:
            raise NotImplementedError("Try a different sampler")

        val_dataloader =  DataLoader(
            dataset=self.train_data,
            sampler=val_sampler,
            batch_size=batch_size,
            num_workers=config.TEST_VAL_DATALOADER_NUM_WORKERS
        )

        return val_dataloader

    def test_data_loader(self, batch_size=config.TEST_VAL_BATCH_SIZE, sampler=config.TEST_VAL_SAMPLER):

        if sampler==config.TEST_VAL_SAMPLER:
            test_sampler = SequentialSampler(self.test_data)

        else:
            raise NotImplementedError("Try a different sampler")

        test_dataloader =  DataLoader(
            dataset=self.train_data,
            sampler=test_sampler,
            batch_size=batch_size,
            num_workers=config.TEST_VAL_DATALOADER_NUM_WORKERS
        )

        return test_dataloader