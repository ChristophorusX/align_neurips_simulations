#!/bin/bash
#SBATCH --partition gpu
#SBATCH --job-name=align_simulation
#SBATCH --out="slurm-%j.out"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL

module load miniconda
conda activate alignment_env

python simulation_align.py
