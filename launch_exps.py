import os
import subprocess

Ks = [4]
penalties = [50,500,5000]
learning_rates = [0.00001, 0.00005, 0.00001]

base_dir = '/cluster/tufts/hugheslab/kheuto01/synth_topk/12_lottery_500_tract_specific'
code_dir = '/cluster/home/kheuto01/code/prob_diff_topk'
epochs=4000
num_components=4
seed=360
threshold=0.45

count = 0
for K in Ks:
    for penalty in penalties:
        for learning_rate in learning_rates:
            
            outdir = os.path.join(base_dir, f'K{K}_penalty{penalty}_lr{learning_rate}')
            
            if not os.path.exists(outdir):
                    os.makedirs(outdir)
                    
            arg_parts = [
                        f"--epochs {epochs}",
                        f"--K {K}",
                        f"--learning_rate {learning_rate}",
                        f"--num_components {num_components}",
                        f"--seed {seed}",
                        f"--outdir {outdir}",
                        f"--penalty {penalty}",
                        f"--threshold {threshold}",
                        ]
            arg_cmd = ' '.join(arg_parts)
            command = (f"code_dir={code_dir} args='{arg_cmd}' sbatch < /cluster/home/kheuto01/code/prob_diff_topk/run_exp.slurm")
            subprocess.run(command, shell=True, check=True)
            count += 1
print(count)
            