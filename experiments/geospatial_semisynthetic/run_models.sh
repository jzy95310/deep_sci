#!/bin/bash
#SBATCH -p carlsonlab-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=sci_geospatial
#SBATCH --output=sci_geospatial_%a.out
#SBATCH --error=sci_geospatial_%a.err
#SBATCH -a 1-6
#SBATCH -c 2
#SBATCH --nice

srun singularity exec --nv --bind /work/zj63 /datacommons/carlsonlab/Containers/multimodal_gp.simg python run_models.py