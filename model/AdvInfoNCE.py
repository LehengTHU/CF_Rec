import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm

class AdvInfoNCE_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)
        self.adv_optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == False], lr=self.adv_lr)

    def modify_saveID(self):
        self.saveID += "_tau" + str(self.model.tau) + '_i_' + str(self.args.adv_interval) + '_ae_' + str(self.args.adv_epochs) + \
            '_al_' + str(self.args.adv_lr) + '_w_' + str(self.args.warm_up_epochs) + '_eta_' + str(self.args.eta_epochs) + \
                '_patience_' + str(self.args.patience) + '_k_neg_' + str(self.args.k_neg)

    def train_one_epoch(self, epoch, optimizer, pbar):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        for batch_i, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[5]
                neg_items_pop = batch[6]

            self.model.train()
            mf_loss, reg_loss = self.model(users, pos_items, neg_items)
            loss = mf_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches]
    

class AdvInfoNCE(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.k_neg = args.k_neg
        self.w_emb_dim = args.w_embed_size
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.model_version = args.model_version

        if(self.model_version == "mlp"): # MLP version
            self.w_emb_dim = 4
            self.u_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.w_emb_dim),
                nn.ReLU())
            self.i_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.w_emb_dim),
                nn.ReLU())
        else: # Embedding version
            self.embed_user_p = nn.Embedding(self.n_users, self.w_emb_dim)
            self.embed_item_p = nn.Embedding(self.n_items, self.w_emb_dim)
            nn.init.xavier_normal_(self.embed_user_p.weight)
            nn.init.xavier_normal_(self.embed_item_p.weight)
        self.freeze_prob(True)
    
    def forward(self, users, pos_items, neg_items):

        #@ Main Branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        #@ Weight Branch
        if(self.model_version == "mlp"):
            users_p_emb = self.u_mlp(userEmb0.detach())
            neg_p_emb = self.i_mlp(negEmb0.detach())
        else:
            users_p_emb = self.embed_user_p(users)
            neg_p_emb = self.embed_item_p(neg_items)

        s_negative = torch.matmul(torch.unsqueeze(users_p_emb, 1), 
                                    neg_p_emb.permute(0, 2, 1)).squeeze(dim=1)

        p_negative = torch.softmax(s_negative, dim=1) # score for negative samples


        # main branch
        # use cosine similarity
        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)

        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)
        
        denominator = numerator + self.k_neg * int(p_negative.shape[1]) * torch.sum(torch.exp(neg_ratings / self.tau)*p_negative, dim = 1) #@ multiply with N

        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        reg_neg_prob = 0.5 * torch.norm(users_p_emb) ** 2 + 0.5 * torch.norm(neg_p_emb) ** 2
        reg_neg_prob = reg_neg_prob / self.batch_size
        reg_loss_prob = self.decay * regularizer

        #@ calculate eta (KL divergence)
        kl_d = (p_negative*torch.log(p_negative/(1/self.neg_sample))).cpu().detach().numpy()
        kl_d = np.sum(kl_d, axis=1)
        # print(kl_d.shape, type(kl_d))
        # print(max(kl_d))
        eta_u_ = {}
        for idx, u in enumerate(list(users.cpu().detach().numpy())):
            kl_d_u = kl_d[idx]
            if u not in eta_u_.keys():
                eta_u_[u] = [kl_d_u]
            else:
                eta_u_[u].append(kl_d_u)

        return ssm_loss, reg_loss, reg_loss_prob, eta_u_, p_negative
    
    def freeze_prob(self, flag):
        if(self.model_version == "mlp"):
            if flag:
                for param in self.u_mlp.parameters():
                    param.requires_grad = False
                for param in self.i_mlp.parameters():
                    param.requires_grad = False
                self.embed_user.requires_grad_(True)
                self.embed_item.requires_grad_(True)
            else:
                for param in self.u_mlp.parameters():
                    param.requires_grad = True
                for param in self.i_mlp.parameters():
                    param.requires_grad = True
                self.embed_user.requires_grad_(False)
                self.embed_item.requires_grad_(False)
        else:
            if flag:
                self.embed_user_p.requires_grad_(False)
                self.embed_item_p.requires_grad_(False)
                self.embed_user.requires_grad_(True)
                self.embed_item.requires_grad_(True)
            else:
                self.embed_user_p.requires_grad_(True)
                self.embed_item_p.requires_grad_(True)
                self.embed_user.requires_grad_(False)
                self.embed_item.requires_grad_(False)