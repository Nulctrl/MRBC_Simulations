#!/bin/bash

#SBATCH --nodes=16
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --job-name=3d_with_ke
#SBATCH --mail-type=END
#SBATCH --mail-user=zn2021@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

cd /scratch/zn2021/Nulctrl1/Nulctrl1
srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 3d.py 

