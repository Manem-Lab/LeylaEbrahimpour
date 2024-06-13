#!/bin/bash
#SBATCH --job-name=rads_cnet
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --mail-type=end,fail
#SBATCH --nodelist=ul-val-pr-gpu[03-04]
#SBATCH --mail-user=leyla.seyed-ebrahimpour.1@ulaval.ca
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_48h
#SBATCH --output=NLST-rads_cnet_bw_0.05.out
#SBATCH --account def-phdes19

module load StdEnv/2020
module load python/3.11.2

# activate virtual env
source /project/166726142/lesee/Synergic-Radiomics/venv/bin/activate
echo "venv activated"

# export JOBDIR=/home/ulaval/lesee/projects/ul-val-prj-def-phdes19/slurm

echo "call python script next"
python -u radiomics-cnet.py > NLST-rads_cnet_bw_0.05.log