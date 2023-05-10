from itertools import product
import os

var3 = [1, 2, 3]
var2 = [0.05]
var1 = ['CEM', 'ICEM', 'MPPI']

i = 0
cwd = os.getcwd()
print(cwd)
#path = os.path.join(cwd, 'code/mbrl/REAI/')
batch_name = 'friction_scripts'
script_path = batch_name
print('script path: ' )
print(script_path)

if not os.path.isdir(script_path):
    os.mkdir(script_path)

for v3 in var3:
    for v2 in var2:
        for v1 in var1:

            batch_file_path = os.path.join(script_path, 'job{}.sh'.format(i))
            with open(batch_file_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("#SBATCH -J {}\n".format(batch_name +'_' + str(i)))
                f.write("#SBATCH --time=4:00:00\n")
                f.write("#SBATCH -N 1\n")
                f.write("#SBATCH -n 1\n")
                f.write("#SBATCH -p gpu\n")
                f.write("#SBATCH --gres=gpu:1\n")
                f.write("#SBATCH -o {}.out\n".format(batch_name +'_' + str(i)))
                f.write("#SBATCH -e {}.err\n".format(batch_name +'_' + str(i)))

                f.write("# Print key runtime properties for records\n")

                f.write("echo Master process running on `hostname`\n")
                f.write("echo Directory is `pwd`\n")
                f.write("echo Starting execution at `date`\n")
                f.write("echo Current PATH is $PATH\n")

                f.write('export APPTAINER_BINDPATH="/gpfs/scratch,/gpfs/data"\n')
                f.write("CONTAINER=/users/gkus/pytorch39.simg\n")

                f.write("#SCRIPT=/users/gkus/code/fourier_neural_operator/experiments/train_fno3d/train_to_fno3d.py\n")
                f.write("SCRIPT=/users/gkus/code/mbrl/REAI/cartpole_script.py\n")
                f.write("#module load mpi/openmpi_4.0.4_gcc\n")
                f.write("#module load gcc/8.3\n")

                f.write("# Run The Job Through Singularity\n")
                f.write("singularity exec --nv $CONTAINER python3 -u $SCRIPT -m  ")
                f.write("optimizer={} optimizer.num_iterations=5 seed={} phys_nn_config=0,1,2,3 physics_model=sindy,cartpole env.track_friction={} env.joint_friction={}".format(v1,v3, v2, 2*v2 ) )
            i = i+1