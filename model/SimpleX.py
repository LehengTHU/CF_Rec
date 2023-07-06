import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from model.base.utils import *

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm

class SimpleX_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def modify_saveID(self):
        self.saveID += "_tau=" + str(self.model.tau) + "w=" + str(self.model.w_neg) + "margin=" + str(self.args.neg_margin)

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            self.model.train()
            if(self.inbatch):
                mf_loss, reg_loss = self.model(users, pos_items)
            else:
                neg_items = batch[5]
                neg_items_pop = batch[6]
                mf_loss, reg_loss = self.model(users, pos_items, neg_items)
            loss = mf_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches]


class SimpleX(AbstractModel):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
    
    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        pos_margin_loss = 1 - pos_ratings
        neg_margin_loss = torch.mean(torch.clamp(neg_ratings - self.margin, 0, 1), dim = -1)
        
        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class SimpleX_batch(AbstractModel):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
    
    def forward(self, users, pos_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        diag_mask = torch.ones_like(ratings_diag).cuda(self.device)-torch.eye(self.batch_size).cuda(self.device)
        
        pos_margin_loss = 1 - ratings_diag
        neg_margin_loss = torch.sum(torch.clamp(ratings - self.margin, 0, 1) * diag_mask, dim = -1) / self.neg_sample

        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

