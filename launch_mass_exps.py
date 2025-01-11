import os
import subprocess
import numpy as np
import sys

# Hyperparameter configurations
Ks = [25, 50, 75]  # Number of top locations to consider
bpr_weights = [0, 30]  # Weights for BPR loss term
nll_weights = [1, 0]  # Weights for negative log-likelihood term
step_sizes = [0.0001, 0.001, 0.01, 0.1]  # Learning rates
perturbed_noises = [0.0001, 0.001, 0.01, 0.1] # Noise levels for perturbed top-k
thresholds = [1]  # BPR thresholds
score_sample_sizes = [ 50]  # Number of samples for score estimation
pert_sample_sizes = [ 50]  # Number of samples for perturbation
exps_context = [('gps', 3), ('asurv', 5)]  # experiments and their corresponding maximum context size
tsteps = ['2monthly']  # timescales to consider

# Fixed parameters
epochs = 8000
seeds = [360]  # Multiple seeds for reproducibility
code_dir = '/cluster/home/fmuenc01/code/prob_diff_topk'

count = 0
for K in Ks:
    # Create base directory for this K value
    base_dir = f"/cluster/tufts/hugheslab/fmuenc01/bird_zero_rand_K{K}_{epochs}"
    
    for bpr_weight in bpr_weights:
        for nll_weight in nll_weights:
            # Skip if both weights are 0 (no learning objective)
            if (bpr_weight == 0 and nll_weight == 0):
                continue
                
            for step_size in step_sizes:
                for seed in seeds:
                    for t, threshold in enumerate(thresholds):
                        for p, perturbed_noise in enumerate(perturbed_noises):
                            for num_score_samples in score_sample_sizes:
                                for num_pert_samples in pert_sample_sizes:
                                    for study, context_size in exps_context:
                                        for timescale in tsteps:
                                            
                                            # Create descriptive output folder name
                                            outfolder = f'K{K}_bw{bpr_weight}_nw{nll_weight}_ss{step_size}_nss{num_score_samples}_nps{num_pert_samples}_seed{seed}'
                                            
                                            # Add perturbation parameters if using BPR
                                            if bpr_weight != 0:
                                                outfolder += f'_sig{perturbed_noise}'
                                                if nll_weight != 0:
                                                    outfolder += f'_tr{threshold}'
                                            else:
                                                # Skip different noise levels if not using BPR
                                                if p > 0:
                                                    continue
                                            
                                            # Skip different thresholds if not using hybrid model
                                            if (bpr_weight == 0 or nll_weight == 0) and t > 0:
                                                continue

                                            outdir = os.path.join(base_dir, outfolder)

                                            # Add corresponding hypers to code dir to load correct data
                                            data_dir = f'data_dir/{study}/{timescale}_ctxtSize{context_size}' # {code_dir}/
                                            outdir += f'_study{study}_timescale{timescale}_ctxtSize{context_size}'
                                            
                                            # Construct command line arguments
                                            arg_parts = [
                                                f"--K {K}",
                                                f"--epochs {epochs}",
                                                f"--step_size {step_size}",
                                                f"--bpr_weight {bpr_weight}",
                                                f"--nll_weight {nll_weight}",
                                                f"--seed {seed}",
                                                f"--outdir {outdir}",
                                                f"--threshold {threshold}",
                                                f"--num_score_samples {num_score_samples}",
                                                f"--num_pert_samples {num_pert_samples}",
                                                f"--perturbed_noise {perturbed_noise}",
                                                f"--data_dir {data_dir}",

                                                "--device cuda"
                                            ]

                                            arg_cmd = ' '.join(arg_parts)
                                            
                                            # Skip if experiment already completed
                                            # if os.path.exists(os.path.join(outdir, 'best_model.pth')):
                                            #     print(f"Skipping completed experiment: {outdir}")
                                            #     continue
                                        
                                            # Launch the job using slurm
                                            command = f"code_dir={code_dir} args='{arg_cmd}' sbatch < run_neg_binom.slurm"  # /cluster/home/fmuenc01/code/prob_diff_topk/
                                            
                                            print(command)
                                            subprocess.run(command, shell=True, check=True)
                                            if count<1:
                                                print(command)
                                            count += 1
                                            sys.exit(0)

                                    

print(f"Launched {count} experiments")