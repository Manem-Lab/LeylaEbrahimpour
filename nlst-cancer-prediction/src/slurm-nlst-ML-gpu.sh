#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu_96h
#SBATCH --job-name=NLST-ML-rads-T0_AddDeltaT1_lasso_Smote
#SBATCH --nodelist=ul-val-pr-cpc03
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=NLST-ML-rads-T0_AddDeltaT1_lasso_Smote.out
#SBATCH --account def-phdes19
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=leyla.seyed-ebrahimpour.1@ulaval.ca



module load StdEnv/2020
module load python/3.11.2

# activate virtual env
source /project/166726142/lesee/Synergic-Radiomics/venv/bin/activate
echo "venv activated"

# export JOBDIR=/home/ulaval/lesee/projects/ul-val-prj-def-phdes19/slurm

echo "call python script next"
python -u ML-CancerPrediction_final_Smote-addDelta-afterLasso.py > NLST-ML-rads-T0_AddDeltaT1_lasso_Smote.log
