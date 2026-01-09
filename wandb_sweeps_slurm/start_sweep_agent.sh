#!/bin/bash
# Run this script on the login node to start the wandb sweep agent
# The agent will submit SLURM jobs for each run

cd /home/andygg98/links/projects/rrg-majewski-ab/andygg98/torch

# Setup Python environment if not already done
if [ ! -d "myenv" ]; then
    echo "Creating virtual environment..."
    module load StdEnv/2023 python/3.10
    python -m venv myenv
    source myenv/bin/activate
    pip install wandb numpy
else
    source myenv/bin/activate
fi

export WANDB_API_KEY="2a1829519497eaab2f05c336830a1d4b0a3a8238"

# Start the wandb agent
# The agent will call run_sweep_job.sh for each run, which submits a SLURM job
wandb agent andygglez-meth/methylation_prediction/3m7tpdtv --count 5