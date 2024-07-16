import os
import subprocess


bpr_weights = [0,500,5000]
nll_weights = [0,1]
step_sizes = [0.1, 0.01, 0.001, 0.0001]

base_dir = '/cluster/tufts/hugheslab/kheuto01/synth_topk_torch_gf_1000/'
code_dir = '/cluster/home/kheuto01/code/prob_diff_topk'
epochs=1000
num_components=4
seed=360
thresholds=[0.55]

count = 0
for bpr_weight in bpr_weights:
    for nll_weight in nll_weights:
        if bpr_weight == 0 and nll_weight == 0:
            continue
        for step_size in step_sizes:
            for threshold in thresholds:
            
                outdir = os.path.join(base_dir, f'bw{bpr_weight}_nw{nll_weight}_ss{step_size}_th{threshold}')
                
                arg_parts = [
                            f"--epochs {epochs}",
                            f"--step_size {step_size}",
                            f"--bpr_weight {bpr_weight}",
                            f"--nll_weight {nll_weight}",
                            f"--seed {seed}",
                            f"--outdir {outdir}",
                            f"--threshold {threshold}",
                            ]


                arg_cmd = ' '.join(arg_parts)
                command = (f"code_dir={code_dir} args='{arg_cmd}' sbatch < /cluster/home/kheuto01/code/prob_diff_topk/run_exp.slurm")
                subprocess.run(command, shell=True, check=True)
                count += 1

print(count)
            