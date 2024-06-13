#!/bin/bash
#SBATCH --job-name=delta-all-rad-os-st1_2_test2
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=120G
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=leyla.seyed-ebrahimpour.1@ulaval.ca
#SBATCH --partition=batch_72h
#SBATCH --output=delta-all-rad-os-st1_2_test2.out
#SBATCH --account def-phdes19

module load StdEnv/2020
module load python/3.11.2

# activate virtual env
source /home/ulaval.ca/lesee/projects/Project2-synergiqc/recurrence/src/venv/bin/activate
echo "venv activated"

export JOBDIR=/home/ulaval/lesee/projects/ul-val-prj-def-phdes19/slurm

echo "call python script next"
python -u ML-delta-OS-all-test2.py>delta-all-rad-os-st1_2_test2.log



#!/bin/bash
 
#SBATCH --job-name=my_python_job
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch_72h
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=my_python_code.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=leyla.seyed-ebrahimpour.1@ulaval.ca

 
module load StdEnv/2020
module load python/3.11.2
 
# activate virtual env 
source """put here the path of the virtual enviornment that you created"+/bin/activate
echo "vrt activated"
 
 
echo "call python script next"
 
python -u my_python_code.py > my_python_code.log