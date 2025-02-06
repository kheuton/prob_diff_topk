#!/usr/bin/env bash
#SBATCH -n 1               # Number of cores
#SBATCH -N 1               # Ensure that all cores are on one machine
#SBATCH -t 1:00:00          # Runtime in D-HH:MM
#SBATCH -p hugheslab           # Partition to submit to
#SBATCH --mem-per-cpu 64000  # Memory (in MB) per cpu
#SBATCH -o /cluster/tufts/hugheslab/kheuto01/slurmlog/out/log_%j.out       # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/kheuto01/slurmlog/err/log_%j.err       # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --gres=gpu:1
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children
##SBATCH --nodelist=s1cmp[001,003,004,005],p1cmp073,p1cmp111

which python
pwd
export PYTHONPATH=$PYTHONPATH:$code_dir
echo $PYTHONPATH
echo "args:"
echo $args

python  torch_opioid_exp_nll_shortcut.py $args