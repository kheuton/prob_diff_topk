import os
import subprocess


bpr_weights = [0, 5000, 100000]
nll_weights = [0,1]
step_sizes = [0.1, 0.5, 0.001]
perturbed_noises = [0.05, 0.01]
initializations = ['bpr', 'nll']
mus = [(10, 30), (10, 50), (30, 50)]

code_dir = '/cluster/home/kheuto01/code/prob_diff_topk'
epochs=4000
num_components=2
seed=360
thresholds=[0.99]
base_dir = f'/cluster/tufts/hugheslab/kheuto01/synth_103050_datainit_{epochs}_comp{num_components}/'

count = 0
for bpr_weight in bpr_weights:
    for nll_weight in nll_weights:
        if bpr_weight == 0 and nll_weight == 0:
            continue
        for step_size in step_sizes:
            for threshold in thresholds:
                for perturbed_noise in perturbed_noises:
                    #for initialization in initializations:
                        for mu1, mu2 in mus:
                
                            outdir = os.path.join(base_dir, f'mu1{mu1}mu2{mu2}_bw{bpr_weight}_nw{nll_weight}_sig{perturbed_noise}_ss{step_size}_th{threshold}')
                            
                            arg_parts = [
                                        f"--epochs {epochs}",
                                        f"--step_size {step_size}",
                                        f"--bpr_weight {bpr_weight}",
                                        f"--nll_weight {nll_weight}",
                                        f"--seed {seed}",
                                        f"--outdir {outdir}",
                                        f"--threshold {threshold}",
                                        f"--num_components {num_components}",
                                        f"--perturbed_noise {perturbed_noise}",
                                        f"--mu1 {mu1}",
                                        f"--mu2 {mu2}"
                                        ]


                            arg_cmd = ' '.join(arg_parts)
                            
                            if os.path.exists(os.path.join(outdir, 'report.png')):
                                print(f"Skipping {outdir}")
                                continue
                            command = (f"code_dir={code_dir} args='{arg_cmd}' sbatch < /cluster/home/kheuto01/code/prob_diff_topk/run_exp.slurm")
                            subprocess.run(command, shell=True, check=True)
                            count += 1

print(count)
            