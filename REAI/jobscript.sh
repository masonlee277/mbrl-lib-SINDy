#!/bin/bash
#SBATCH -J mbrl_1
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o mbrl_1.out
#SBATCH -e mbrl_1.err

# Print key runtime properties for records

echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

export APPTAINER_BINDPATH="/gpfs/scratch,/gpfs/data"
CONTAINER=/users/gkus/pytorch39.simg

#SCRIPT=/users/gkus/code/fourier_neural_operator/experiments/train_fno3d/train_to_fno3d.py
SCRIPT=/users/gkus/code/mbrl/REAI/cartpole_script.py
#module load mpi/openmpi_4.0.4_gcc
#module load gcc/8.3

# Run The Job Through Singularity
singularity exec --nv $CONTAINER python3 -u $SCRIPT -m  optimizer=CEM,ICEM,MPPI seed=1,2,3 phys_nn_config=0,1,2,3 physics_model=sindy,cartpole 
