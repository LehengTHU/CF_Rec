import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm


class InvCF_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        running_pop_mf_loss, running_pop_reg_loss, running_sub_reg_loss, running_sub_mf_loss, running_disc_loss = 0, 0, 0, 0, 0
        

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            self.model.train()
            if(self.inbatch):
                mf_loss, reg_loss, pop_mf_loss, pop_reg_loss, sub_mf_loss, sub_reg_loss, disc_loss = self.model(users, pos_items, users_pop, pos_items_pop)
            else:
                neg_items = batch[5]
                neg_items_pop = batch[6]
                mf_loss, reg_loss, pop_mf_loss, pop_reg_loss, sub_mf_loss, sub_reg_loss, disc_loss = self.model(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop)
            
            if self.args.need_distance == 1:
                loss = mf_loss + reg_loss + pop_mf_loss + pop_reg_loss + sub_mf_loss + sub_reg_loss - disc_loss
            else:
                loss = mf_loss + reg_loss + pop_mf_loss + pop_reg_loss + sub_mf_loss + sub_reg_loss         

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            running_pop_mf_loss += pop_mf_loss.detach().item()
            running_pop_reg_loss += pop_reg_loss.detach().item()
            running_sub_mf_loss += sub_mf_loss.detach().item()
            running_sub_reg_loss += sub_reg_loss.detach().item()
            if self.args.need_distance == 1:
                running_disc_loss += disc_loss.detach().item()
            num_batches += 1

        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches,\
                running_pop_mf_loss / num_batches, running_pop_reg_loss / num_batches,
                running_sub_mf_loss / num_batches, running_sub_reg_loss / num_batches,
                running_disc_loss / num_batches]


class InvCF(AbstractModel):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
    
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.n_factors = int(args.n_factors)
        self.distype = args.distype
        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.need_distance = args.need_distance
        self.kernel = args.kernel

        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
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

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = torch.zeros(1).cuda(self.device)
       
        ui_embeddings = torch.cat([cor_u_embeddings, cor_i_embeddings],0)
        # TODO
        ui_factor_embeddings = torch.split(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors-1):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i+1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)

        return cor_loss
    
    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = torch.sum(torch.square(X),1,keepdims=True)
            w = torch.bmm(X.unsqueeze(1), X.unsqueeze(-1)).squeeze(-1)
            D = torch.sqrt(torch.maximum(r - 2 * w + r.t(), torch.tensor(0.0)) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - torch.mean(D,dim=0,keepdims=True)-torch.mean(D,dim=1,keepdims=True) \
                + torch.mean(D)

            return D
        
        def _create_distance_covariance(D1,D2):
            #calculate distance covariance between D1 and D2
            n_samples = D1.shape[0]
            dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
            # dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2)) / n_samples)
            return dcov
        
        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1,D2)
        dcov_11 = _create_distance_covariance(D1,D1)
        dcov_22 = _create_distance_covariance(D2,D2)
        
        #calculate the distance correlation
        dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, torch.tensor(0.0))) + 1e-10)


        #return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def distance_loss(self, users, pos_items, users_pop, pos_items_pop):
        if self.distype == 'l1':
            return self.l_norm(users, pos_items, users_pop, pos_items_pop, 1)
        elif self.distype == 'l2':
            return self.l_norm(users, pos_items, users_pop, pos_items_pop, 2)
        elif self.distype == 'dcor':
            return self.dcor_loss(users, pos_items, users_pop, pos_items_pop)
        elif self.distype == 'mmd':
            return self.mmd_loss(users, users_pop) + self.mmd_loss(pos_items, pos_items_pop)
            print("Error distance")

    def l_norm(self, users, pos_items, users_pop, pos_items_pop, p_value):
        
        user_loss = torch.norm(( users - users_pop ), p = p_value)
        item_loss = torch.norm(( pos_items - pos_items_pop ), p = p_value)
        return (user_loss + item_loss)/self.batch_size

    def dcor_loss(self, users, pos_items, users_pop, pos_items_pop):
        
        user_loss = self.create_cor_loss(users, users_pop)
        pos_loss = self.create_cor_loss(pos_items, pos_items_pop)
        #neg_loss = self.create_cor_loss(neg_items, neg_items_pop)

        return user_loss + pos_loss # + neg_loss

    def mmd_loss(self,x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
        XX, YY, XY = (torch.zeros(xx.shape).cuda(self.device),
                  torch.zeros(xx.shape).cuda(self.device),
                  torch.zeros(xx.shape).cuda(self.device))
         
        if self.kernel == "multiscale":
            
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if self.kernel == "rbf":
        
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        
        

        return torch.mean(XX + YY - 2. * XY)
        
    def infonce_loss(self, users, pos_items, neg_items, userEmb0, posEmb0, negEmb0, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
   
    def switch_concat(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, \
                            userEmb0, posEmb0, negEmb0, userEmb0_p, posEmb0_p, negEmb0_p):

        users_ori = torch.cat((users_pop, users), -1)
        pos_items_ori = torch.cat((pos_items_pop, pos_items), -1)
        neg_items_ori = torch.cat((neg_items, neg_items_pop), -1)
        users_ori_0 = torch.cat((userEmb0_p, userEmb0), -1)
        pos_items_ori_0 = torch.cat((posEmb0_p, posEmb0), -1)
        neg_items_ori_0 = torch.cat((negEmb0_p, negEmb0), -1)

        mf_loss, reg_loss = self.infonce_loss(users_ori, pos_items_ori, neg_items_ori, users_ori_0, pos_items_ori_0, neg_items_ori_0)

        random_order = torch.randperm(pos_items_pop.size()[0])
        pos_items_pop_new = pos_items_pop[random_order]
        pos_items_new = torch.cat((pos_items_pop_new, pos_items), -1)
        users_pop_new = users_pop[random_order]
        users_new = torch.cat((users_pop_new, users), -1)

        users_new_0 = torch.cat((userEmb0_p[random_order], userEmb0), -1)
        pos_items_new_0 = torch.cat((posEmb0_p[random_order], posEmb0), -1)

        mf_loss_new, reg_loss_new = self.infonce_loss(users_new, pos_items_new,  neg_items_ori, users_new_0, pos_items_new_0, neg_items_ori_0)

        return mf_loss + mf_loss_new, reg_loss + reg_loss_new
       
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        #all_users_p, all_items_p = self.compute_p()

        #users_pop = all_users_p[users]
        #pos_items_pop = all_users_p[pos_items]
        #neg_items_pop = all_users_p[neg_items]

        users_pop = userEmb0_p
        pos_items_pop = posEmb0_p
        neg_items_pop = negEmb0_p

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        
        mf_loss, reg_loss = self.infonce_loss(users, pos_items, neg_items, userEmb0, posEmb0, negEmb0)
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, neg_items_pop, userEmb0_p, posEmb0_p, negEmb0_p)
        mf_loss_new, reg_loss_new = self.switch_concat(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, \
                                                        userEmb0, posEmb0, negEmb0, userEmb0_p, posEmb0_p, negEmb0_p)
            
        disc_loss = 0
        if self.need_distance == 1:
            disc_loss = self.lambda2 * self.distance_loss(users, pos_items, users_pop, pos_items_pop)
        
        return mf_loss, reg_loss, self.lambda1 * pop_mf_loss, self.lambda1 * pop_reg_loss, self.lambda3 * mf_loss_new, self.lambda3 * reg_loss_new, disc_loss


class InvCF_batch(InvCF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample =  self.batch_size-1
    
    def infonce_loss(self, users, pos_items, userEmb0, posEmb0, batch_size = 1024):

        users_emb = F.normalize(users, dim=1)
        pos_emb = F.normalize(pos_items, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2  +  0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
   
    def switch_concat(self, users, pos_items, users_pop, pos_items_pop, userEmb0, posEmb0, userEmb0_p, posEmb0_p):

        users_ori = torch.cat((users_pop, users), -1)
        pos_items_ori = torch.cat((pos_items_pop, pos_items), -1)
        users_ori_0 = torch.cat((userEmb0_p, userEmb0), -1)
        pos_items_ori_0 = torch.cat((posEmb0_p, posEmb0), -1)

        mf_loss, reg_loss = self.infonce_loss(users_ori, pos_items_ori, users_ori_0, pos_items_ori_0)

        random_order = torch.randperm(pos_items_pop.size()[0])
        pos_items_pop_new = pos_items_pop[random_order]
        pos_items_new = torch.cat((pos_items_pop_new, pos_items), -1)
        users_pop_new = users_pop[random_order]
        users_new = torch.cat((users_pop_new, users), -1)

        users_new_0 = torch.cat((userEmb0_p[random_order], userEmb0), -1)
        pos_items_new_0 = torch.cat((posEmb0_p[random_order], posEmb0), -1)

        mf_loss_new, reg_loss_new = self.infonce_loss(users_new, pos_items_new, users_new_0, pos_items_new_0)

        return mf_loss + mf_loss_new, reg_loss + reg_loss_new
        
        return mf_loss, reg_loss
    
    def forward(self, users, pos_items, users_pop, pos_items_pop):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_users_p[pos_items]
        #users_pop = userEmb0_p
        #pos_items_pop = posEmb0_p
        
        users = all_users[users]
        pos_items = all_items[pos_items]
        
        mf_loss, reg_loss = self.infonce_loss(users, pos_items, userEmb0, posEmb0)
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, userEmb0_p, posEmb0_p)
        mf_loss_new, reg_loss_new = self.switch_concat(users, pos_items, users_pop, pos_items_pop, \
                                                    userEmb0, posEmb0, userEmb0_p, posEmb0_p, )
            
        disc_loss = 0
        if self.need_distance == 1:
            disc_loss = self.lambda2 * self.distance_loss(users, pos_items, users_pop, pos_items_pop)
        
        return mf_loss, reg_loss, self.lambda1 * pop_mf_loss, self.lambda1 * pop_reg_loss, self.lambda3 * mf_loss_new, self.lambda3 * reg_loss_new, disc_loss     

