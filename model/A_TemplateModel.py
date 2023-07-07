import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm

'''

An example of template model

Listed are possible functions that can be implemented in the template model.

'''

class template_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def modify_saveID(self):
        raise NotImplementedError
    
    def set_optimizer(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        raise NotImplementedError
    

class template(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau

    def compute(self):
        raise NotImplementedError
    
    def forward(self, users, pos_items, neg_items):
        raise NotImplementedError