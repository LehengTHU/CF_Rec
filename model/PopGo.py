import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm


class PopGo_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.freeze_eoch = args.freeze_epoch

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        running_loss1, running_loss2=0,0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for _, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, _, neg_items, neg_items_pop  = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

            self.model.train()
            
            loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = self.model(users,pos_items,neg_items,users_pop,pos_items_pop,neg_items_pop)
            
            if epoch<self.freeze_epoch:
                loss =  loss2 + reg_loss_freeze
            else:
                self.model.freeze_pop()
                loss = loss1 + reg_loss_norm
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss1 += loss1.detach().item()
            running_loss2 += loss2.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_loss +=  loss1.detach().item()+loss2.detach().item()+reg_loss.detach().item()
            num_batches += 1

        return [running_loss / num_batches,
                running_loss1 / num_batches, running_loss2 / num_batches, running_reg_loss / num_batches]

class PopGo(AbstractModel):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        all_users, all_items = self.compute()

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        users_pop = self.embed_user_pop(users_pop)
        pos_items_pop = self.embed_item_pop(pos_items_pop)
        neg_items_pop = self.embed_item_pop(neg_items_pop)

        tiled_usr=torch.tile(users,[1,self.neg_sample]).reshape([-1,self.emb_dim])
        tiled_usr_pop=torch.tile(users_pop,[1,self.neg_sample]).reshape([-1,self.emb_dim])
        user_n2=torch.norm(users,p=2,dim=1)
        user_pop_n2=torch.norm(users_pop,p=2,dim=1)
        tiled_usr_n2=torch.norm(tiled_usr,p=2,dim=1)
        tiled_usr_pop_n2=torch.norm(tiled_usr_pop,p=2,dim=1)
        pos_item_n2=torch.norm(pos_items,p=2,dim=1)
        neg_item_n2=torch.norm(neg_items,p=2,dim=1)
        neg_item_pop_n2=torch.norm(neg_items_pop,p=2,dim=1)
        pos_item_pop_n2=torch.norm(pos_items_pop,p=2,dim=1)

        pos_item_pop_prod=torch.sum(torch.mul(users_pop,pos_items_pop),dim=1)
        neg_item_pop_prod=torch.sum(torch.mul(tiled_usr_pop,neg_items_pop),dim=1)
        pos_item_prod=torch.sum(torch.mul(users,pos_items),dim=1)
        neg_item_prod=torch.sum(torch.mul(tiled_usr,neg_items),dim=1)

        # option 1: sigmoid dot-product
        # pos_item_score=tf.sigmoid(pos_item_prod)
        # neg_item_score=tf.sigmoid(neg_item_prod)
        # pos_item_pop_score=tf.sigmoid(pos_item_pop_prod)/self.tau2
        # neg_item_pop_score=tf.sigmoid(neg_item_pop_prod)/self.tau2


        # option 2: cosine similarity
        pos_item_score=pos_item_prod/user_n2/pos_item_n2
        neg_item_score=neg_item_prod/tiled_usr_n2/neg_item_n2
        pos_item_pop_score=pos_item_pop_prod/user_pop_n2/pos_item_pop_n2/self.tau2
        neg_item_pop_score=neg_item_pop_prod/tiled_usr_pop_n2/neg_item_pop_n2/self.tau2

        # pure infonce loss
        #pos_item_score_mf_exp=torch.exp(pos_item_score/self.tau1)
        #neg_item_score_mf_exp=torch.sum(torch.exp(torch.reshape(neg_item_score/self.tau,[-1,self.neg_sample])),dim=1)
        #loss_mf=torch.mean(torch.negative(torch.log(pos_item_score_mf_exp/(pos_item_score_mf_exp+neg_item_score_mf_exp))))


        neg_item_pop_score_exp=torch.sum(torch.exp(torch.reshape(neg_item_pop_score,[-1,self.neg_sample])),dim=1)
        pos_item_pop_score_exp=torch.exp(pos_item_pop_score)
        loss2=self.w_lambda*torch.mean(torch.negative(torch.log(pos_item_pop_score_exp/(pos_item_pop_score_exp+neg_item_pop_score_exp))))

        weighted_pos_item_score=torch.mul(pos_item_score,torch.sigmoid(pos_item_pop_prod))/self.tau1
        weighted_neg_item_score=torch.mul(neg_item_score,torch.sigmoid(neg_item_pop_prod))/self.tau1

        neg_item_score_exp=torch.sum(torch.exp(torch.reshape(weighted_neg_item_score,[-1,self.neg_sample])),dim=1)
        pos_item_score_exp=torch.exp(weighted_pos_item_score)
        loss1=(1-self.w_lambda)*torch.mean(torch.negative(torch.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp))))

        regularizer1 = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop) ** 2 + 0.5 * torch.norm(pos_items_pop) ** 2 + 0.5 * torch.norm(neg_items_pop) ** 2 
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm
    
    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


