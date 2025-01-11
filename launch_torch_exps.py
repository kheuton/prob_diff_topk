import os
import subprocess


bpr_weights = [0,30]
nll_weights = [1, 0]
step_sizes = [0.1,]
perturbed_noises = [0.05, 0.01]
init_indices = range(20)
thresholds = [0.5,  1]
#mus = [(10, 30), (10, 50), (30, 50)]

code_dir = '/cluster/home/fmuenc01/code/prob_diff_topk'
epochs=4000
num_components=7
seeds=[360]

base_dir = f'/cluster/tufts/hugheslab/fmuenc01/frontier_{epochs}_comp{num_components}/'

count = 0

for bpr_weight in bpr_weights:
    for nll_weight in nll_weights:
        if (bpr_weight == 0 and nll_weight == 0):
            continue
        for step_size in step_sizes:
            for t, threshold in enumerate(thresholds):
                for p, perturbed_noise in enumerate(perturbed_noises):
                    for init_idx in init_indices:
                        #for mu1, mu2 in mus:
                        #for seed in seeds:
                            outfolder = f'bw{bpr_weight}_nw{nll_weight}_ss{step_size}'
                            if bpr_weight != 0:
                                outfolder += f'_sig{perturbed_noise}'
                                if nll_weight != 0:
                                    outfolder += f'_tr{threshold}'
                            else:
                                # bpr weight is 0, skip if this is second noise
                                if p > 0:
                                    continue

                            if bpr_weight ==0 or nll_weight == 0:
                                # not hybrid models, skip threshold
                                if t>0:
                                    continue
                            outfolder += f'_init{init_idx}'

                            outdir = os.path.join(base_dir, outfolder)
                            
                            arg_parts = [
                                        f"--epochs {epochs}",
                                        f"--step_size {step_size}",
                                        f"--bpr_weight {bpr_weight}",
                                        f"--nll_weight {nll_weight}",
                                        f"--init_idx {init_idx}",
                                        f"--outdir {outdir}",
                                        f"--threshold {threshold}",
                                        f"--num_components {num_components}",
                                        f"--perturbed_noise {perturbed_noise}",
                                        f"--data frontier",
                                        ]


                            arg_cmd = ' '.join(arg_parts)
                            
                            if os.path.exists(os.path.join(outdir, 'report.png')):
                                print(f"Skipping {outdir}")
                                continue
                            command = (f"code_dir={code_dir} args='{arg_cmd}' sbatch < /cluster/home/fmuenc01/code/prob_diff_topk/run_exp.slurm")
                            print(command)
                            subprocess.run(command, shell=True, check=True)
                            count += 1
                            sys.exit(0)


print(count)
            