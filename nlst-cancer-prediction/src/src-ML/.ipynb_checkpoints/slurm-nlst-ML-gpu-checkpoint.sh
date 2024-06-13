#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu_48h
#SBATCH --job-name=NLST-ML-addDelta1_lasso_Smote-KT_ND-GB
#SBATCH --nodelist=ul-val-pr-cpc[02-03],ul-val-pr-gpu04
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=./Smote-addDelta/NLST-ML-addDelta1_lasso_Smote-combat-KT_ND-GB.out
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
python -u ./Smote-addDelta/ML-CancerPrediction_final_Smote-addDelta-afterLasso2.py > ./Smote-addDelta/NLST-ML-addDelta1_lasso_Smote-combat-KT_ND-GB.log
