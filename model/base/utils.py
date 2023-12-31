import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
random.seed(101)
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
#from scipy.linalg import svd
import itertools
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
# from evaluator import ProxyEvaluator
import collections
import os

def merge_user_list(user_lists):
    out = collections.defaultdict(list)
    # Loop over each user list
    for user_list in user_lists:
        # Loop over each user in the user list
        for key, item in user_list.items():
            out[key] = out[key] + item
    return out


def merge_user_list_no_dup(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item
    
    for key in out.keys():
        out[key]=list(set(out[key]))
    return out


def save_checkpoint(model, epoch, checkpoint_dir, buffer, max_to_keep=10):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)
    buffer.append(filename)
    if len(buffer)>max_to_keep:
        os.remove(buffer[0])
        del(buffer[0])

    return buffer


def restore_checkpoint(model, checkpoint_dir, device, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0,

    epoch_list = []

    regex = re.compile(r'\d+')

    for cp in cp_files:
        epoch_list.append([int(x) for x in regex.findall(cp)][0])

    epoch = max(epoch_list)

   
    if not force:
        print("Which epoch to load from? Choose in range [0, {})."
              .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end = '')
        # inp_epoch = int(input())
        inp_epoch = epoch
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0,
    else:
        print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(0, epoch):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename, map_location = str(device))

    try:
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
              .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch


def restore_best_checkpoint(epoch, model, checkpoint_dir, device):
    """
    Restore the best performance checkpoint
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename, map_location = str(device))

    model.load_state_dict(checkpoint['state_dict'])
    print("=> Successfully restored checkpoint (trained for {} epochs)"
          .format(checkpoint['epoch']))

    return model


def clear_checkpoint(checkpoint_dir):
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def evaluation(args, data, model, epoch, base_path, evaluator, name="valid"):
    # Evaluate with given evaluator

    ret, _ = evaluator.evaluate(model)

    n_ret = {"recall": ret[1], "hit_ratio": ret[5], "precision": ret[0], "ndcg": ret[3], "mrr":ret[4], "map":ret[2]}

    perf_str = name+':{}'.format(n_ret)
    print(perf_str)
    with open(base_path + 'stats.txt', 'a') as f:
        f.write(perf_str + "\n")
    # Check if need to early stop (on validation)
    is_best=False
    early_stop=False
    if name=="valid":
        if ret[1] > data.best_valid_recall:
            data.best_valid_epoch = epoch
            data.best_valid_recall = ret[1]
            data.patience = 0
            is_best=True
        else:
            data.patience += 1
            if data.patience >= args.patience:
                print_str = "The best performance epoch is % d " % data.best_valid_epoch
                print(print_str)
                early_stop=True

    return is_best, early_stop, n_ret


def Item_pop(args, data, model):

    for K in range(5):

        eval_pop = ProxyEvaluator(data, data.train_user_list, data.pop_dict_list[K], top_k=[(K+1)*10],
                                   dump_dict=merge_user_list([data.train_user_list, data.valid_user_list]))

        ret, _ = eval_pop.evaluate(model)

        print_str = "Overlap for K = % d is % f" % ( (K+1)*10, ret[1] )

        print(print_str)

        with open('stats.txt', 'a') as f:
            f.write(print_str + "\n")


def ensureDir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_grp_view(data,grp_idx):
    n=len(grp_view)
    split_data=[{} for _ in range(n)]

    for key,item in data.items():
        for it in item:
            if key not in split_data[grp_idx[it]].keys():
                split_data[grp_idx[it]][key]=[]
            split_data[grp_idx[it]][key].append(it)
    return split_data


def checktensor(tensor):
    t=tensor.detach().cpu().numpy()
    if np.max(np.isnan(t)):        
        idx=np.argmax(np.isnan(t))
        return idx
    else:
        return -1

def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


grads = {}
def save_grad(name):
    def hook(grad):
        torch.clamp(grad, -1, 1)
        grads[name] = grad
    return hook

"""
It takes in the embeddings of users and items, the popularity of users and items, and the data, and
plots the embeddings of users and items in 3D space, and plots the angular distribution of the
embeddings of items. 

The embeddings of users and items are plotted in 3D space, and the angular distribution of the
embeddings of items are plotted in 2D space. 

:param items: the item embeddings
:param users: the user embeddings
:param data: the data object
:param p_item: the popularity of each item
:param p_user: the popularity of each user
:param name: the name of the dataset
"""
def visulization(items,users,data,p_item,p_user,name):
    test_ood_user_list=data.test_ood_user_list
    test_id_user_list=data.test_id_user_list
    train_user_list=data.train_user_list

    def split_grp_view(data,grp_idx):
        n=len(grp_view)
        split_data=[collections.defaultdict(list) for _ in range(n)]

        for key,item in data.items():
            for it in item:
                if key not in split_data[grp_idx[it]].keys():
                    split_data[grp_idx[it]][key]=[]
                split_data[grp_idx[it]][key].append(it)
        return split_data

    pop_sorted=np.sort(p_item)
    n_items=p_item.shape[0]

    n_groups=3
    grp_view=[]
    for grp in range(n_groups):
        split=int((n_items-1)*(grp+1)/n_groups)
        grp_view.append(pop_sorted[split])
    #print("group_view:",grp_view)
    idx=np.searchsorted(grp_view,p_item)

    pop_group=[[] for _ in range(n_groups)]
    for i in range(n_items):
        pop_group[idx[i]].append(i)

    eval_test_ood_split=split_grp_view(test_ood_user_list,idx)
    eval_test_id_split=split_grp_view(test_id_user_list,idx)
    eval_train_split=split_grp_view(train_user_list,idx)

    pop_users=p_user.tolist()

    u_pop_sorted=np.sort(p_user)
    print(u_pop_sorted[-10:])



    fig = plt.figure(constrained_layout=True,figsize=(12,6))


    def plot_embed(ax1,ax2,idx):
        u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]
        x1 = np.cos(u)*np.sin(v)
        y1 = np.sin(u)*np.sin(v)
        z1 = np.cos(v)
        ax1.plot_wireframe(x1, y1, z1, color="0.5",linewidth=0.1)
        user_idx=pop_users.index(idx)
        m_user=users[user_idx]
        target=np.array([1,-1,1])
        r_theta=np.arccos(np.dot(m_user,target)/(np.linalg.norm(m_user)*np.linalg.norm(target)))
        axis=np.cross(m_user,target)
        R=get_rotation_matrix(axis,r_theta)
        grp_theta=[]
        grp_r=[]
        sizes=[10,10,10]

        cmap_b = 'b'
        cmap_r = 'r'
        cmaps=[cmap_b,cmap_r]

        norm = plt.Normalize(vmin=-3, vmax=3)

        all_sampled=set([])
        all_pos=set([])
        for i,grp in enumerate(pop_group):
            sampled_group=set(np.random.choice(np.array(grp),50,replace=False).tolist())
            if user_idx in eval_test_id_split[i].keys():
                for item in eval_test_id_split[i][user_idx]:
                    sampled_group.add(item)
                    all_pos.add(item)
            for item in eval_train_split[i][user_idx]:
                sampled_group.add(item)
                all_pos.add(item)
            if user_idx in eval_test_ood_split[i].keys():
                for item in eval_test_ood_split[i][user_idx]:
                    sampled_group.add(item)
                    all_pos.add(item)
            
            all_sampled=all_sampled.union(sampled_group)
            

        all_neg=all_sampled.difference(all_pos)
        #print(all_neg)
        all_pos=np.array(list(all_pos),dtype=int)
        all_neg=np.array(list(all_neg),dtype=int)
        nor = plt.Normalize(vmin=-3, vmax=3)
        r=np.linalg.norm(target)

        lab=["neg","pos"]
        for i,idx in enumerate([all_neg,all_pos]):
            g_item=items[idx]
            g_item=np.matmul(g_item,R.T)
            norm=np.linalg.norm(g_item,axis=1)
            x=g_item[:,0]/norm#*r
            y=g_item[:,1]/norm#*r
            z=g_item[:,2]/norm#*r

            for j in range(len(idx)):
                ax1.plot([0,g_item[j][0]/norm[j]],[0,g_item[j][1]/norm[j]],[0,g_item[j][2]/norm[j]],color = cmaps[i],alpha=0.1)

            ax1.scatter(x, y, z, c = cmaps[i], marker =".",s=10,label=lab[i])

        
        #print("V^{T}",V_transpose)

        ax1.scatter(target[0]/r, target[1]/r, target[2]/r, c = 'g', marker ="*",s=120,label="user")
        ax1.plot([0,target[0]/r],[0,target[1]/r],[0,target[2]/r],color = 'g',alpha=0.1)
        ax1.legend()

        

        all_items=set([i for i in range(n_items)])
        all_neg=all_items.difference(all_pos)
        all_neg=np.array(list(all_neg),dtype=int)

        grp=["(neg):","(pos):"]
        txt=""

        for i,idx in enumerate([all_neg,all_pos]):
            g_item=items[idx]
            g_item=np.matmul(g_item,R.T)
            norm=np.linalg.norm(g_item,axis=1)
            cos=np.arccos(np.matmul(target,g_item.T)/norm/r)
            me=float(np.mean(cos))
            me=round(me,3)
            if i==1:
                txt="mean angle"+grp[i]+str(me)+"\n"+txt
            else:
                txt="mean angle"+grp[i]+str(me)+txt
            
            ax2.hist(cos,50,range=[0,np.pi],color=cmaps[i],weights=np.zeros_like(cos) + 1. / cos.size,edgecolor='black',alpha=0.6)
            mi_x,ma_x=ax2.get_xlim()
            mi_y,ma_y=ax2.get_ylim()
        ax2.text(mi_x+(ma_x-mi_x)*0.45, mi_y+(ma_y-mi_y)*0.75,txt , style ='italic') 
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='pos')
        blue_patch = mpatches.Patch(color='blue', alpha=0.6,label='neg')
        ax2.legend(handles=[red_patch,blue_patch])
        
    
    pops=[205,30,10]


    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax2 = fig.add_subplot(gs[2,0:2])
    ax1.set_xticks([-1,-0.5,0,0.5,1])
    ax1.set_yticks([-1,-0.5,0,0.5,1])
    ax1.set_zticks([-1,-0.5,0,0.5,1])
    ax1.grid(False)

    plot_embed(ax1,ax2,pops[0])
    #ax1.set_title("High Pop User(p=205)")
    #ax2.set_title("Angular Distribution(High Pop)")

    plt.savefig(name+"high_pop_"+str(pops[0])+".png",bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax3 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax4 = fig.add_subplot(gs[2,0:2])
    ax3.set_xticks([-1,-0.5,0,0.5,1])
    ax3.set_yticks([-1,-0.5,0,0.5,1])
    ax3.set_zticks([-1,-0.5,0,0.5,1])
    ax3.grid(False)
    plot_embed(ax3,ax4,pops[1])
    #ax3.set_title("Mid Pop User(p=30)")
    #ax4.set_title("Angular Distribution(Mid Pop)")

    plt.savefig(name+"mid_pop_"+str(pops[1])+".png",bbox_inches='tight')
    plt.close()



    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax5 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax6 = fig.add_subplot(gs[2,0:2])
    ax5.set_xticks([-1,-0.5,0,0.5,1])
    ax5.set_yticks([-1,-0.5,0,0.5,1])
    ax5.set_zticks([-1,-0.5,0,0.5,1])
    ax5.grid(False)
    plot_embed(ax5,ax6,pops[2])
    #ax5.set_title("Low Pop User(p=10)")
    #ax6.set_title("Angular Distribution(Low Pop)")

    plt.savefig(name+"low_pop_"+str(pops[2])+".png",bbox_inches='tight')
    plt.close()

def seed_torch(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def visualize_and_save_log(file_dir, dataset_name, show=False):
    # 逐行读取file_dir文件, 只保留
    if(dataset_name == "tencent_synthetic"):
        pass
    else:
        id_recall, id_ndcg, ood_recall, ood_ndcg = [], [], [], []

        with open(file_dir, 'r') as f:
            # count = 0
            for line in f:
                line = line.split(' ')
                if("valid" in line[0]):
                    id_recall.append(float(line[1][:-1]))
                    id_ndcg.append(float(line[7][:-1]))
                if("test_ood" in line[0]):
                    ood_recall.append(float(line[1][:-1]))
                    ood_ndcg.append(float(line[7][:-1]))

        epochs = list(range(0, len(id_recall)))
        epochs = [i*5 for i in epochs]
        # 定义表格
        result = pd.DataFrame({'epochs': epochs, 'id_recall': id_recall, 'ood_recall': ood_recall, 'id_ndcg': id_ndcg, 'ood_ndcg': ood_ndcg})
        # df是除了最后一行的所有行
        df = result.iloc[:-1, :]

        fig=plt.figure()
        x = df.epochs
        y1 = df.id_recall
        y2 = df.ood_recall
        print(max(y1), max(y2), 1.1*max(y1), 1.1*max(y2))
        #ax1显示y1  ,ax2显示y2 
        ax1=fig.subplots()
        ax2=ax1.twinx()    #使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
        ax1.plot(x,y1,'g-', label='id_recall')
        ax2.plot(x,y2,'b--', label='ood_recall')
        # 坐标轴范围
        ax1.set_ylim(min(y1), 1.15*(max(y1)-min(y1))+min(y1))
        ax2.set_ylim(min(y2), 1.15*(max(y2)-min(y2))+min(y2))

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('id_recall')
        ax2.set_ylabel('ood_recall')
        # legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        base_path = file_dir[:-9]
        save_path = base_path + "/train_log.png"
        plt.savefig(save_path)
        if(show):
            plt.show()
        save_path = base_path + "/train_log.csv"
        result.to_csv(save_path, index=False)