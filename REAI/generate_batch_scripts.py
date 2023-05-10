from itertools import product
import os

var3 = [1, 2, 3]
var2 = [0, 0.05]
var1 = ['CEM', 'ICEM', 'MPPI']

i = 0
cwd = os.getcwd()
print(cwd)
#path = os.path.join(cwd, 'code/mbrl/REAI/')
batch_name = 'pretraining_length'
script_path = batch_name
print('script path: ' )
print(script_path)

vm = ['sindy','cartpole']
vpnnc = [1,2,0,3]
vf = [0.0, 0.05]


if not os.path.isdir(script_path):
    os.mkdir(script_path)

for v1 in vf:
    for v2 in vm:
        for v3 in vpnnc:
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
                    f.write("singularity exec --nv $CONTAINER python3 -u $SCRIPT -m  pretrain_trial_length=10,20,50,100 ")
                    f.write("optimizer=CEM,ICEM,MPPI seed=1,2,3 phys_nn_config={} physics_model={} env.track_friction={} env.joint_friction={}".format(v3,v2, v1, 2*v1 ) )
                i = i+1