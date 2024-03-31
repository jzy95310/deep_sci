#!/bin/bash
#SBATCH -p scavenger-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=sci_geospatial
#SBATCH --output=sci_geospatial_%a.out
#SBATCH --error=sci_geospatial_%a.err
#SBATCH -a 1-14
#SBATCH -c 2
#SBATCH --nice

export WANDB__SERVICE_WAIT=300

srun singularity exec --nv --bind /work/zj63 /datacommons/carlsonlab/Containers/multimodal_gp.simg python run_models.py