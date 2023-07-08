#!/bin/bash

# Define the hyperparameters and their respective values

param_values=(0.2 0.22 0.24 0.26 0.28 0.30)
# param_values=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)

length=${#param_values[@]}

# Function to run the model training for a given parameter combination
run_model() {
    local param=$1
    
    # Run your model training command with the given parameters
    # Replace the following command with your actual model training command
    nohup python main.py --cuda 0 --n_layers 2 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --train_norm --pred_norm --modeltype InfoNCE --tau $param --dataset yahoo.new --candidate --saveID eval_tau &> logs/0708_eval_yahoo_tau.log &
    # nohup python main.py --cuda 1 --n_layers 2 --batch_size 2048 --lr $param --neg_sample 1 --infonce 0 --modeltype LightGCN --dataset yahoo.new --candidate --saveID lr &> logs/0708_eval_yahoo_lgn_lr.log &

}

# Launch parallel processes for grid search
for ((i=0; i<$length; i++)); do
       
    param=${param_values[i]}
       
    run_model $param  &
done

# Wait for all the processes to finish
wait

echo "Start Searching!"
