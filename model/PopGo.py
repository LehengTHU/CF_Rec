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
        self.freeze_epoch = args.freeze_epoch

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        running_loss1, running_loss2=0,0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for _, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]

            users, pos_items, users_pop, pos_items_pop, _  = batch[0], batch[1], batch[2], batch[3], batch[4]


            if(self.inbatch):
                self.model.train()
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = self.model(users, pos_items, users_pop, pos_items_pop)
            else:
                neg_items = batch[5]
                neg_items_pop = batch[6]

                self.model.train()
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = self.model(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop)
                  
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
        self.decay = args.decay
        self.w_lambda = args.w_lambda
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)


    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        return users, items

        
    def infonce_loss(self, users, pos_items, neg_items, userEmb0, posEmb0, negEmb0, tau, batch_size = 102):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / tau)
        denominator = torch.sum(torch.exp(ratings / tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 


    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]    

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, neg_items_pop, userEmb0_p, posEmb0_p, negEmb0_p, self.tau2)
        loss2=self.w_lambda * pop_mf_loss

        # loss1的处理
        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_item_score = torch.sum(users_emb*pos_emb, dim = -1)
        neg_item_score = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        
        pos_item_pop_prod = torch.sum(users_pop*pos_items_pop, dim = -1) 
        neg_item_pop_prod = torch.matmul(torch.unsqueeze(users_pop, 1), neg_items_pop.permute(0, 2, 1)).squeeze(dim=1)

        pos_ratings = torch.mul(pos_item_score,torch.sigmoid(pos_item_pop_prod))
        neg_ratings = torch.mul(neg_item_score,torch.sigmoid(neg_item_pop_prod))

        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
 
        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        loss1 = (1-self.w_lambda)*ssm_loss

        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer1 = regularizer1 / self.batch_size
        reg_loss_norm=self.decay * (regularizer1)

        reg_loss = reg_loss_norm + pop_reg_loss

        return loss1, loss2, reg_loss, pop_reg_loss, reg_loss_norm
    
    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

class PopGo_batch(PopGo):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample =  self.batch_size-1

    
    def infonce_loss(self, users, pos_items, userEmb0, posEmb0, tau, batch_size = 1024):

        users_emb = F.normalize(users, dim=1)
        pos_emb = F.normalize(pos_items, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / tau)
        denominator = torch.sum(torch.exp(ratings / tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2  +  0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 

    def forward(self, users, pos_items, users_pop, pos_items_pop):

        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
    
        users = all_users[users]
        pos_items = all_items[pos_items]

        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, userEmb0_p, posEmb0_p, self.tau2)
        loss2=self.w_lambda * pop_mf_loss

        # loss1的处理
        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        
        pos_item_score = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))

        pos_item_pop_prod = torch.matmul(users_pop, torch.transpose(pos_items_pop, 0, 1))
        
        ratings = torch.mul(pos_item_score,torch.sigmoid(pos_item_pop_prod))

        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        loss1 = (1-self.w_lambda)*ssm_loss

        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 
        regularizer1 = regularizer1 / self.batch_size
        reg_loss_norm=self.decay * (regularizer1)

        reg_loss = reg_loss_norm + pop_reg_loss

        return loss1, loss2, reg_loss, pop_reg_loss, reg_loss_norm
    
    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)




