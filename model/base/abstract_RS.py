import numpy as np
import torch
import torch.nn as nn
from data.data import Data
from evaluator import ProxyEvaluator
from model.base.utils import *
import datetime

# define the abstract class for recommender system
class AbstractRS(nn.Module):
    def __init__(self, args, special_args) -> None:
        super(AbstractRS, self).__init__()

        # basic information
        self.args = args
        self.special_args = special_args
        self.device = torch.device(args.cuda)
        self.test_only = args.test_only

        self.Ks = args.Ks
        self.patience = args.patience
        self.modeltype = args.modeltype
        self.neg_sample = args.neg_sample
        self.inbatch = self.args.infonce == 1 and self.args.neg_sample == -1
    
        # basic hyperparameters
        self.n_layers = args.n_layers
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.epoch
        self.verbose = args.verbose

        # load the data
        self.dataset_name = args.dataset
        try:
            print('from model.'+ args.modeltype + ' import ' + args.modeltype + '_Data')
            exec('from model.'+ args.modeltype + ' import ' + args.modeltype + '_Data') # load special dataset
            self.data = eval(args.modeltype + '_Data(args)') 
        except:
            print("no special dataset")
            self.data = Data(args) # load data from the path
        
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.train_user_list = self.data.train_user_list
        self.valid_user_list = self.data.valid_user_list
        # = torch.tensor(self.data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(self.data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(self.data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = self.data.user_pop_max
        self.item_pop_max = self.data.item_pop_max 

        # load the model
        self.running_model = args.modeltype + '_batch' if self.inbatch else args.modeltype
        exec('from model.'+ args.modeltype + ' import ' + self.running_model) # import the model first
        self.model = eval(self.running_model + '(args, self.data)') # initialize the model with the graph
        self.model.cuda(self.device)

        # preparing for saving
        self.preperation_for_saving(args, special_args)
        
        # preparing for evaluation
        self.not_candidate_dict = self.data.get_not_candidate() # load the not candidate dict
        self.evaluators, self.eval_names = self.get_evaluators(self.data, self.not_candidate_dict) # load the evaluators

    # the whole pipeline of the training process
    def execute(self):

        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) # restore the checkpoint

        # train the model if not test only
        if not self.test_only:
            print("start training") 
            self.train()
            # test the model
            print("start testing")
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.device)
        
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d" % self.data.best_valid_epoch
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        n_rets = {}
        for i,evaluator in enumerate(self.evaluators[:]):
            _, __, n_ret = evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, evaluator, self.eval_names[i])
            n_rets[self.eval_names[i]] = n_ret

        self.document_hyper_params_results(self.base_path, n_rets)

    # define the training process
    def train(self) -> None:
        # TODO
        self.set_optimizer() # get the optimizer
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            # print(self.model.embed_user.weight)
            if self.flag: # early stop
                break
            # All models
            t1=time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)

        visualize_and_save_log(self.base_path +'stats.txt', self.dataset_name)

    #! must be implemented by the subclass
    def train_one_epoch(self, epoch):
        raise NotImplementedError
    
    def preperation_for_saving(self, args, special_args):
        self.formatted_today=datetime.date.today().strftime('%m%d') + '_'

        tn = '1' if args.train_norm else '0'
        pn = '1' if args.pred_norm else '0'
        self.train_pred_mode = 't' + tn + 'p' + pn

        self.saveID = self.formatted_today + args.saveID + '_' + self.train_pred_mode + "_Ks=" + str(args.Ks) + '_patience=' + str(args.patience)\
            + "_n_layers=" + str(args.n_layers) + "_batch_size=" + str(args.batch_size)\
                + "_neg_sample=" + str(args.neg_sample) + "_lr=" + str(args.lr) 
        
        for arg in special_args:
            print(arg, getattr(args, arg))
            self.saveID += "_" + arg + "=" + str(getattr(args, arg))
        
        self.modify_saveID()

        if self.n_layers > 0 and self.modeltype != "LightGCN":
            self.base_path = './weights/{}/{}-LGN/{}'.format(self.dataset_name, self.running_model, self.saveID)
        else:
            self.base_path = './weights/{}/{}/{}'.format(self.dataset_name, self.running_model, self.saveID)
        self.checkpoint_buffer=[]
        ensureDir(self.base_path)

    def modify_saveID(self):
        pass

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)

    def document_running_loss(self, losses:list, epoch, t_one_epoch, prefix=""):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = prefix + 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + 'stats.txt','a') as f:
                f.write(perf_str+"\n")
    
    def document_hyper_params_results(self, base_path, n_rets):
        overall_path = '/'.join(base_path.split('/')[:-1]) + '/'
        hyper_params_results_path = overall_path + self.dataset_name + '_' + self.modeltype + '_' + self.args.saveID + '_hyper_params_results.csv'

        results = {'notation': self.formatted_today, 'train_pred_mode':self.train_pred_mode, 'best_epoch': max(self.data.best_valid_epoch, self.start_epoch), 'max_epoch': self.max_epoch, 'Ks': self.Ks, 'n_layers': self.n_layers, 'batch_size': self.batch_size, 'neg_sample': self.neg_sample, 'lr': self.lr}
        for special_arg in self.special_args:
            results[special_arg] = getattr(self.args, special_arg)

        for k, v in n_rets.items():
            if('test_id' not in k):
                for metric in ['ndcg', 'recall', 'hit_ratio']:
                    results[k + '_' + metric] = v[metric]
        frame_columns = list(results.keys())
        # load former xlsx
        if os.path.exists(hyper_params_results_path):
            # hyper_params_results = pd.read_excel(hyper_params_results_path)
            hyper_params_results = pd.read_csv(hyper_params_results_path)
        else:
            # 用results创建一个新的dataframe
            hyper_params_results = pd.DataFrame(columns=frame_columns)

        hyper_params_results = hyper_params_results.append(results, ignore_index=True)
        # to csv
        hyper_params_results.to_csv(hyper_params_results_path, index=False)
        # hyper_params_results.to_excel(hyper_params_results_path, index=False)


    
    # define the evaluation process
    def eval_and_check_early_stop(self, epoch):
        self.model.eval()

        for i,evaluator in enumerate(self.evaluators):
            is_best, temp_flag, n_ret = evaluation(self.args, self.data, self.model, epoch, self.base_path, evaluator, self.eval_names[i])
            
            if is_best:
                checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
            
            # early stop?
            if temp_flag:
                self.flag = True

        self.model.train()
    
    # load the checkpoint
    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
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

            if self.args.clear_checkpoints:
                print("Clear checkpoint")
                clear_checkpoint(checkpoint_dir)
                return model, 0,

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
    
    def restore_best_checkpoint(self, epoch, model, checkpoint_dir, device):
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
    
    def get_evaluators(self, data, not_candidate_dict=None, pop_mask=None):
        #if not self.args.pop_test:
        K_value = self.args.Ks
        if(self.dataset_name == "tencent_synthetic"):
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[K_value])
            eval_test_ood_1 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_1,top_k=[K_value],\
                                    dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_2,data.test_ood_user_list_3]))
            eval_test_ood_2 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_2,top_k=[K_value],\
                                    dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_1,data.test_ood_user_list_3]))
            eval_test_ood_3 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_3,top_k=[K_value],\
                                    dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_1,data.test_ood_user_list_2]))
                        
        elif  self.args.candidate:
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list,not_candidate_dict]))
            eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list,not_candidate_dict]))
            eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list,not_candidate_dict]))
        else: 
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[K_value])  
            eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]))
            eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[K_value],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]))
       
        if(self.dataset_name == "tencent_synthetic" or self.dataset_name == "kuairec_ood"):
            evaluators=[eval_valid, eval_test_ood_1, eval_test_ood_2, eval_test_ood_3]
            eval_names=["valid","test_ood_1", "test_ood_2", "test_ood_3"]
        else:
            evaluators=[ eval_valid,eval_test_id, eval_test_ood]
            eval_names=["valid","test_id", "test_ood" ]

        return evaluators, eval_names

    

