import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm
from data.data import Data

import pickle
import random
import scipy.sparse as sp
import os


class UltraGCN_pop_Data(Data):
    def __init__(self, args):
        super().__init__(args)
        

    def pload(self, path):
        with open(path, 'rb') as f:
            res = pickle.load(f)
        print('load path = {} object'.format(path))
        return res

    def pstore(self, x, path):
        with open(path, 'wb') as f:
            pickle.dump(x, f)
        print('store object in path = {} ok'.format(path))

    def get_ii_constraint_mat(self, train_mat, num_neighbors, ii_diagonal_zero = False):
        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)	# I * I
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long(), res_sim_mat.float()

    def add_special_model_attr(self, args):

        train_mat = None
        self.constraint_mat = None
        self.i_constraint_mat = None
        self.ii_neighbor_mat = None

        train_data_temp = []
              
        for i in range(len(self.trainUser)):
            train_data_temp.append([self.trainUser[i], self.trainItem[i]])
            
        train_mat = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        for x in train_data_temp:
            train_mat[x[0], x[1]] = 1.0
            # construct degree matrix for graphmf

        items_D = np.sum(train_mat, axis = 0).reshape(-1)
        users_D = np.sum(train_mat, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        self.constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                            "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
            
        ii_cons_mat_path = self.path + '_ii_constraint_mat'
        ii_neigh_mat_path = self.path + '_ii_neighbor_mat'
            
        if os.path.exists(ii_cons_mat_path):
            self.ii_constraint_mat = self.pload(ii_cons_mat_path)
            self.ii_neighbor_mat = self.pload(ii_neigh_mat_path)
        else:
            self.ii_neighbor_mat, self.ii_constraint_mat = self.get_ii_constraint_mat(train_mat, args.ii_neighbor_num)
            
            self.pstore(self.ii_constraint_mat, ii_cons_mat_path)
            self.pstore(self.ii_neighbor_mat, ii_neigh_mat_path)


class UltraGCN_pop_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.freeze_epoch = args.freeze_epoch

    def train_one_epoch(self, epoch):
        running_loss, _, running_reg_loss, num_batches = 0, 0, 0, 0
        running_loss1, running_loss2=0,0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for _, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, _  = batch[0], batch[1], batch[2], batch[3], batch[4]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
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


class UltraGCN_pop(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.w4 = args.w4
        self.negative_weight = args.negative_weight
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.lambda_2 = args.lambda_2

        self.constraint_mat = data.constraint_mat
        self.ii_constraint_mat = data.ii_constraint_mat
        self.ii_neighbor_mat = data.ii_neighbor_mat
        self.device = data.device

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
     
    def infonce_loss(self, users, pos_items, neg_items, tau, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / tau)
        denominator = torch.sum(torch.exp(ratings / tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(users_emb) ** 2 + 0.5 * torch.norm(pos_emb) ** 2 + 0.5 ** torch.norm(neg_emb)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

    def get_omegas(self, users, pos_items, neg_items):
        
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).cuda(self.device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).cuda(self.device)
        
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).cuda(self.device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).cuda(self.device)

        weight = torch.cat((pow_weight, neg_weight))
        return weight
    
    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        
        user_embeds = self.embed_user(users)
        pos_embeds = self.embed_item(pos_items)
        neg_embeds = self.embed_item(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).cuda(self.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).cuda(self.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()
         
    def cal_loss(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):
        
        user_embeds = self.embed_user(users)
        pos_embeds = self.embed_item(pos_items)
        neg_embeds = self.embed_item(neg_items)

        regularizer = 0.5 * torch.norm(user_embeds) ** 2 + 0.5 * torch.norm(pos_embeds) ** 2 + 0.5 * torch.norm(neg_embeds) ** 2
        regularizer = regularizer / self.batch_size
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        pos_item_pop_prod = (users_pop * pos_items_pop).sum(dim=-1)
        pos_scores = torch.mul(pos_scores,torch.sigmoid(pos_item_pop_prod))
        neg_item_pop_prod = torch.matmul(torch.unsqueeze(users_pop, 1), neg_items_pop.permute(0, 2, 1)).squeeze(dim=1)
        neg_scores = torch.mul(neg_scores,torch.sigmoid(neg_item_pop_prod))

        neg_labels = torch.zeros(neg_scores.size()).cuda(self.device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels,  reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).cuda(self.device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels,  reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum(), regularizer
    
    def cal_loss_I(self, users, pos_items):
        
        neighbor_embeds = self.embed_item(self.ii_neighbor_mat[pos_items].cuda(self.device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].cuda(self.device)     # len(pos_items) * num_neighbors
        user_embeds = self.embed_user(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        return loss.sum()
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):
        
        users_pop = self.embed_user_pop(users_pop)
        pos_items_pop = self.embed_item_pop(pos_items_pop)
        neg_items_pop = self.embed_item_pop(neg_items_pop)
        loss2, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, neg_items_pop, self.tau2)

        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss1, regularizer = self.cal_loss(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop)
        loss1 += self.lambda_ * self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss1 += self.lambda_2 * self.cal_loss_I(users, pos_items)
       
        reg_loss_norm = self.gamma * regularizer

        reg_loss = reg_loss_norm + pop_reg_loss

        return loss1, loss2, reg_loss, pop_reg_loss, reg_loss_norm
    
