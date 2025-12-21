#!/bin/bash
#SBATCH --time=0-02:00
#SBATCH --job-name=torch.training
#SBATCH -c 8
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --account=def-majewski_gpu
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=andy.garciagonzalez@mail.mcgill.ca



module load StdEnv/2023 python/3.10 cuda/12.6

export WANDB_MODE=offline
export WANDB_API_KEY="2a1829519497eaab2f05c336830a1d4b0a3a8238"

# El Ko
python -m venv myenv &> /dev/null
pip install --no-index torch lightning wandb matplotlib &> /dev/null

python lightning.training.py

# /home/andygg98/environments/pytorch/bin/