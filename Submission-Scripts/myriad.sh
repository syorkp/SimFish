#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# Request V100 node only (A100 causes error)
#$ -ac allow=EF

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:0:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=40G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=30G

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/zcbtspi/Scratch/

# Change into temporary directory to run work
cd /home/zcbtspi/
rm -r .nv
cd /home/zcbtspi/Scratch/SimFish/

# load the cuda module (in case you are running a CUDA program)
module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/3.7
module load cuda/10.0.130/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-10.0
module load tensorflow/2.0.0/gpu-py37
module load ffmpeg/4.1/gnu-4.9.2


# Run the application - the line below is just a random example.
python3 run.py "$runconfig"
