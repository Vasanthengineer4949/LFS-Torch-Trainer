import lfs.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler

class LFSDataLoader():

    def __init__(self, train_dataset, val_dataset=None, test_dataset=None, *args, **kwargs):

        '''
        A class which contains the functions to create dataloader for the train, test and val datasets.
        
        Input:
        train_dataset: Training data to be used
        val_dataset: Validation data to be used. Default: None
        test_dataset: Testing data to be used. Defult: None
        '''

        super().__init__(*args, **kwargs)
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset

    def train_data_loader(self, batch_size=config.TRAIN_BATCH_SIZE, sampler=config.TRAIN_SAMPLER):

        '''
        A function which is used to return the training dataloader object
        
        Input:
        train_data: Training data to sample and batch. From self
        batch_size: int : Size of each batch. Default : 64
        sampler: Sampler to be used for sampling the data. Default : random
        
        Returns the train_dataloader object
        '''
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

        '''
        A function which is used to return the validation dataloader object
        
        Input:
        val_data: Validation data to sample and batch. From self
        batch_size: int : Size of each batch. Default : 32
        sampler: Sampler to be used for sampling the data. Default : sequential
        
        Returns the val_dataloader object
        '''

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

        '''
        A function which is used to return the testing dataloader object
        
        Input:
        test_data: Testing data to sample and batch. From self
        batch_size: int : Size of each batch. Default : 32
        sampler: Sampler to be used for sampling the data. Default : sequential
        
        Returns the test_dataloader object
        '''

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