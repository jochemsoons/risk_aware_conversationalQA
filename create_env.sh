#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=ExampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

srun conda env create -f jeroen.yml