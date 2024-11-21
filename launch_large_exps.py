import os
import subprocess

Ks = [100]
bpr_weights = [0, 30]
nll_weights = [0, 1]
step_sizes = [0.001, 0.0001]
perturbed_noises = [0.001, 0.01, 0.1]
#init_indices = range(20)
thresholds = [1, 0.8]

init_indices = range(20)
#mus = [(10, 30), (10, 50), (30, 50)]

code_dir = '/cluster/home/kheuto01/code/prob_diff_topk'
epochs=4000
#num_components=7
seeds=[360]
score_sample_sizes = [50, ]
pert_sample_sizes = [50,]


count = 0
for K in Ks:
    base_dir = f'/cluster/tufts/hugheslab/kheuto01/large_synth_band_heavysteep_lots_init_{K}_{epochs}/'
    for bpr_weight in bpr_weights:
        for nll_weight in nll_weights:
            if (bpr_weight == 0 and nll_weight == 0):
                continue
            for step_size in step_sizes:
                for t, threshold in enumerate(thresholds):
                    for p, perturbed_noise in enumerate(perturbed_noises):
                        for init_idx in init_indices:
                            for num_score_samples in score_sample_sizes:
                                for num_pert_samples in pert_sample_sizes:
                            #for init_idx in init_indices:
                                #for mu1, mu2 in mus:
                                #for seed in seeds:
                                    outfolder = f'K{K}_bw{bpr_weight}_nw{nll_weight}_ss{step_size}_nss{num_score_samples}_nps{num_pert_samples}'
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
                                                f"--K {K}",
                                                f"--epochs {epochs}",
                                                f"--step_size {step_size}",
                                                f"--bpr_weight {bpr_weight}",
                                                f"--nll_weight {nll_weight}",
                                                f"--init_idx {init_idx}",
                                                f"--outdir {outdir}",
                                                f"--threshold {threshold}",
                                                f"--num_score_samples {num_score_samples}",
                                                f"--num_pert_samples {num_pert_samples}",
                                                f"--perturbed_noise {perturbed_noise}",
                                                ]


                                    arg_cmd = ' '.join(arg_parts)
                                    
                                    if os.path.exists(os.path.join(outdir, 'report.png')):
                                        print(f"Skipping {outdir}")
                                        continue
                                    command = (f"code_dir={code_dir} args='{arg_cmd}' sbatch < /cluster/home/kheuto01/code/prob_diff_topk/run_large_exp.slurm")
                                    subprocess.run(command, shell=True, check=True)
                                    count += 1


print(count)
            