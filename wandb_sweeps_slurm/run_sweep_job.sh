#!/bin/bash
# This script is called by wandb agent and submits a SLURM job for each sweep run

# Create a unique job script for this run
TIMESTAMP=$(date +%s)
JOB_SCRIPT="/home/andygg98/links/projects/rrg-majewski-ab/andygg98/torch/job_${TIMESTAMP}.slurm"

cat > ${JOB_SCRIPT} << 'EOF'
#!/bin/bash
#SBATCH --time=0-04:00
#SBATCH --job-name=torch.sweep
#SBATCH -c 8
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --account=def-majewski_gpu
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andy.garciagonzalez@mail.mcgill.ca

module load StdEnv/2023 python/3.10 cuda/12.6

export WANDB_API_KEY="2a1829519497eaab2f05c336830a1d4b0a3a8238"

# Activate virtual environment
cd /home/andygg98/links/projects/rrg-majewski-ab/andygg98/torch
source myenv/bin/activate

# Run the training script - wandb will provide the hyperparameters
srun python lightning.training.py --npz chr19.filtered.npz \
                                    --num_workers 8 \
                                    --epochs 350 \
                                    --accelerator gpu
EOF

# Submit the job
sbatch ${JOB_SCRIPT}

# Clean up the temporary job script (optional - comment out if you want to keep them)
# rm ${JOB_SCRIPT}