#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=48:00:00

# Set partition to small mode
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# Set output file
#SBATCH --output=%x.%j

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=sam.york.pink@gmail.com

# Change working directory?
#$ -wd /SimFish

set runconfig "$runconfig"

echo starting

module load python/anaconda3

#module load cuda/9.2
module load cuda/10.2
module load ffmpeg/4.2.2

source activate simfishenv

cd SimFish

# run the application
python3 run.py "$runconfig"
