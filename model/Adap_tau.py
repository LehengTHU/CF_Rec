import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.abstract_model import AbstractModel
from model.base.abstract_RS import AbstractRS
from tqdm import tqdm
from scipy.special import lambertw

from torch_scatter import scatter

class Adap_tau_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.cnt_lr = args.cnt_lr

        train_cf = torch.LongTensor(np.array([np.array(self.data.trainUser), np.array(self.data.trainItem)], np.int32).T)
        self.loss_per_user = None
        self.loss_per_ins = None
        # prepare for tau_0
        self.pos = train_cf.to(self.device)
        nu = scatter(torch.ones(len(train_cf), device=self.device), self.pos[:, 0], dim=0, reduce='sum') #每个用户交互的物品数量
        nu_thresh = torch.quantile(nu, 0.2)
        judgeid_torch = (nu > nu_thresh)
        [self.useid_torch, ] = torch.where(judgeid_torch > 0)
        [self.yid_torch ,] = torch.where(judgeid_torch[self.pos[:,0]]>0)

    def modify_saveID(self):
        self.saveID += "_tau=" + str(self.model.tau) + 'warm_up=' + str(self.args.cnt_lr)

    def train_one_epoch(self, epoch):
        
        losses_train = []
        tau_maxs = []
        tau_mins = []
        losses_emb = []
        hits = 0

        if epoch >= self.cnt_lr:
            user_emb_cos, item_emb_cos = self.model.compute()

            user_emb_cos = F.normalize(user_emb_cos, dim=-1).detach()
            item_emb_cos = F.normalize(item_emb_cos, dim=-1).detach()

            pos_scores = (user_emb_cos[self.pos[:, 0]] * item_emb_cos[self.pos[:, 1]]).sum(dim=-1)
            pos_u_torch = pos_scores[self.yid_torch].mean() #miu+
            # pos_var_torch = pos_scores[yid_torch].var()
            ev_mean_torch = item_emb_cos.mean(dim=0, keepdim=True) # item 做平均之后的一个平均的item值
            allu_torch = (user_emb_cos[self.useid_torch] @ ev_mean_torch.t()).view(-1)
            au_torch = allu_torch.mean() #miu
            can_torch = np.log(len(self.useid_torch) * self.data.n_items)
            a_torch = 1e-10
            c_torch = 2 * (np.log(0.5)+can_torch-np.log(len(self.yid_torch)))
            b_torch = - (pos_u_torch - au_torch)
            w_0 = c_torch / (-2 * b_torch)
        else:
            w_0 = torch.tensor(1/self.model.tau)
        
        print("current tau_0 is {}".format(1/w_0))

        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            if(batch_i == 0):
                train_cf_ = users
            else:
                train_cf_ = torch.cat((train_cf_, users))

            self.model.train()
            if(self.inbatch):
                mf_loss, mf_loss_, reg_loss, tau = self.model(users, pos_items, self.loss_per_user, w_0=w_0, s=batch_i)
            else:
                neg_items = batch[5]
                neg_items_pop = batch[6]
                mf_loss, mf_loss_, reg_loss, tau = self.model(users, pos_items, neg_items, self.loss_per_user, w_0=w_0, s=batch_i)
            loss = mf_loss + reg_loss

            tau_maxs.append(tau.max().item())
            tau_mins.append(tau.min().item())
            losses_train.append(mf_loss_)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            num_batches += 1
        
        losses_train = torch.cat(losses_train, dim=0)
        # loss_per_user = scatter(losses_train, train_cf_, dim=0, reduce='mean').detach() #算每个user的loss
        loss_per_user = scatter(losses_train, train_cf_, dim=0, reduce='mean')

        perf_str = ' TAU_0:{:.4}, TAU_u:{:.4} {:.4}'.format(
                        1/w_0, 1/np.mean(tau_mins), 1/np.mean(tau_maxs)) + ' W_0:{:.4}, W_u:{:.4} {:.4}'.format(
                        w_0, np.mean(tau_mins), np.mean(tau_maxs))

        #@ 表现写入txt文件
        with open(self.base_path + 'stats_{}.txt'.format(self.args.saveID),'a') as f:
            f.write(perf_str+"\n")

        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches]
    
class Adap_tau(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.adap_tau_beta = args.adap_tau_beta

        self.lambertw_table = torch.FloatTensor(lambertw(np.arange(-1, 1002, 1e-4))).to(self.device)
        self.register_buffer("memory_tau", torch.full((self.data.n_users,), 1 / 0.10))
    
    def _loss_to_tau(self, x, x_all):
        t_0 = x_all #t_0其实是reverse的
        if x is None:
            tau = t_0 * torch.ones_like(self.memory_tau, device=self.device) #如果还没有x，则直接返回t_0
        else:
            base_laberw = torch.mean(x) # x是loss
            laberw_data = torch.clamp((x - base_laberw) / self.adap_tau_beta, #t2是beta
                                    min=-np.e ** (-1), max=1000) # 这个是lambertw中间的值
            laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
            tau = (t_0 * torch.exp(-laberw_data)).detach()

        return tau

    def _update_tau_memory(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau = x

    def forward(self, users, pos_items, neg_items, loss_per_user, w_0, s):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        if s == 0 and w_0 is not None:
            tau_user = self._loss_to_tau(loss_per_user, w_0)
            self._update_tau_memory(tau_user)

        w = torch.index_select(self.memory_tau, 0, users).detach()

        numerator = torch.exp(pos_ratings * w_0)
        # denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        denominator = torch.sum(torch.exp(ratings * w.unsqueeze(1)), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        numerator_ = torch.exp(pos_ratings)
        # denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        denominator_ = torch.sum(torch.exp(ratings), dim = 1)
        ssm_loss_ = torch.negative(torch.log(numerator_/denominator_))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, ssm_loss_, reg_loss, w