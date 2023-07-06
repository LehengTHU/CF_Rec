#!/bin/bash

# Define the hyperparameters and their respective values
# tau
param_values=(1e-2 5e-2 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

length=${#param_values[@]}

# Function to run the model training for a given parameter combination
run_model() {
    local param=$1
    
    # Run your model training command with the given parameters
    # Replace the following command with your actual model training command
    nohup python main.py --candidate --tau $param --modeltype INFONCE --dataset yahoo.new --n_layers 2 --batch_size 2048 --lr 5e-4 --neg_sample 128 --infonce 1 --dsc 0704 &> logs/0704_test_CF_Rec.log &
}

# Launch parallel processes for grid search
for ((i=0; i<$length; i++)); do
       
    param=${param_values[i]}
       
    run_model $param  &
done

# Wait for all the processes to finish
wait

echo "Tuning completed!"
