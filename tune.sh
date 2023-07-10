#!/bin/bash

# Define the hyperparameters and their respective values

# tau
param_values=(0.2)
# param_values=(0.1 0.2 0.3 0.4)
# param_values=(0.2 0.22 0.24 0.26 0.28 0.30)
# lr
# param_values=(5e-5 1e-4 5e-4 1e-3)

# eta_epochs
# param_values=(6 7 8 9 10)

# batch_size
# param_values=(512 1024 2048 4096)

# neg_sample
# param_values=(32 64 128 256 512)

length=${#param_values[@]}

# Function to run the model training for a given parameter combination
run_model() {
    local param=$1
    # Run your model training command with the given parameters
    # Replace the following command with your actual model training command
    # yahoo MF lr
    # nohup python main.py --cuda 0 --n_layers 0 --batch_size 2048 --lr $param --neg_sample 1 --infonce 0 --modeltype LightGCN --dataset yahoo.new --candidate --saveID eval_lr &> logs/0708_eval_yahoo_mf_lr.log &

    # yahoo InfoNCE tau
    # nohup python main.py --cuda 0 --n_layers 2 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype InfoNCE --tau $param --dataset yahoo.new --candidate --saveID eval_tau &> logs/0708_eval_yahoo_tau.log &
    
    # yahoo InfoNCE neg_sample
    # nohup python main.py --cuda 0 --n_layers 2 --batch_size 2048 --lr 5e-4 --neg_sample $param --infonce 1 --train_norm --pred_norm --modeltype InfoNCE --tau 0.24 --dataset yahoo.new --candidate --saveID eval_n_neg &> logs/0708_eval_yahoo_tau.log &
    
    # yahoo LGN lr
    # nohup python main.py --cuda 1 --n_layers 2 --batch_size 2048 --lr $param --neg_sample 1 --infonce 0 --modeltype LightGCN --dataset yahoo.new --candidate --saveID lr &> logs/0708_eval_yahoo_lgn_lr.log &
    # nohup python main.py --cuda 1 --n_layers 2 --batch_size 2048 --lr $param --neg_sample 1 --infonce 0 --modeltype LightGCN --dataset yahoo.new --Ks 20 --saveID lr &> logs/0708_eval_yahoo_lgn_lr.log &

    # KuaiRand MF lr candidate
    # nohup python main.py --cuda 1 --n_layers 0 --batch_size 2048 --lr $param --neg_sample 1 --infonce 0 --modeltype LightGCN --dataset KuaiRand --candidate --Ks 5 --saveID eval_lr_candidate &> logs/0708_eval_kuairand_mf_lr_candidate.log &   

    # KuaiRand InfoNCE tau
    nohup python main.py --cuda 0 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype InfoNCE --tau $param --dataset KuaiRand --Ks 5 --candidate --saveID eval_tau &> logs/0710_eval_kuairand_tau_$param.log &
    # no candidate 
    # nohup python main.py --cuda 1 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype InfoNCE --tau $param --dataset KuaiRand --Ks 20 --saveID eval_tau_no_candidate &> logs/0710_eval_kuairand_tau_no_candidate.log &

    # KuaiRand InfoNCE neg_sample
    # nohup python main.py --cuda 1 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample $param --infonce 1 --train_norm --pred_norm --modeltype InfoNCE --tau 0.2 --dataset KuaiRand --Ks 5 --candidate --saveID eval_n_neg &> logs/0710_eval_kuairand_n_neg.log &

    # KuaiRand AdvInfoNCE adv_lr
    # nohup python main.py --cuda 1 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype AdvInfoNCE --tau 0.3 --eta_epochs 5 --adv_lr $param --dataset KuaiRand --Ks 5 --candidate --saveID eval_adv_epoch &> logs/0708_eval_kuairand_adv_epoch_$param_$pid.log &

    # KuaiRand AdvInfoNCE eta_epochs
    # nohup python main.py --cuda 1 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype AdvInfoNCE --tau 0.3 --eta_epochs $param --adv_lr 1e-4 --dataset KuaiRand --Ks 5 --candidate --saveID eval_eta &> logs/0708_eval_kuairand_eta_$param_$pid.log &
    # nohup python main.py --cuda 1 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype AdvInfoNCE --tau 0.3 --eta_epochs $param --adv_lr 1e-4 --dataset KuaiRand --Ks 5 --candidate --saveID eval_eta_add &> logs/0708_eval_kuairand_eta_add_$param.log &
    # nohup python main.py --cuda 0 --n_layers 0 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype AdvInfoNCE --tau 0.3 --eta_epochs $param --adv_lr 1e-4 --dataset KuaiRand --Ks 20 --saveID eval_eta_no_candidate &> logs/0710_eval_kuairand_eta_$param.log &


    # KuaiRand LGN lr
    # nohup python main.py --cuda 1 --n_layers 2 --batch_size 2048 --lr $param --neg_sample 1 --infonce 0 --modeltype LightGCN --dataset KuaiRand --Ks 20 --saveID eval_lr &> logs/0708_eval_kuairand_lgn_lr.log &
}

# Launch parallel processes for grid search
for ((i=0; i<$length; i++)); do
       
    param=${param_values[i]}
       
    run_model $param  &
done

# Wait for all the processes to finish
wait

echo "Start Searching!"
