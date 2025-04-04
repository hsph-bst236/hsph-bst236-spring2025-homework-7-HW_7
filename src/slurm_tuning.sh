#!/bin/bash
#SBATCH --job-name=hyperparam_tuning
#SBATCH --output=sbatch_logs/hyperparam_tuning.log
#SBATCH --error=sbatch_logs/hyperparam_tuning.err
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00

BASH_PATH=$(which bash)

# Define the learning rates to test (Change this along with ntasks as needed)
learning_rates=(1e-4 1e-3 5e-3)

# Run the tuning script in parallel for each learning rate
for lr in "${learning_rates[@]}"; do
    srun --exclusive -N 1 -n 1 \
        $BASH_PATH -c "export WANDB_API_KEY=TODO && \
                venv/bin/python3 src/run_hyperparam_tuning.py --learning_rate $lr" &
done


# Wait for all background jobs to finish
wait

echo "Hyperparameter tuning completed."
