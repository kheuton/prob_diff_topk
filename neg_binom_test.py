import os
import argparse

def get_test_config(local=True):
    """
    Generate a single test configuration for the Negative Binomial model.
    
    Args:
        local (bool): If True, uses settings appropriate for local testing
                     If False, uses settings for cluster testing
    
    Returns:
        str: Command line arguments as a single string
    """
    if local:
        # Minimal configuration for quick local testing
        config = {
            "K": 10,                    # Small K for faster computation
            "epochs": 2,                # Just a couple epochs to test the loop
            "step_size": 0.001,
            "bpr_weight": 1.0,
            "nll_weight": 1.0,
            "seed": 42,
            "threshold": 0.55,
            "num_score_samples": 5,     # Small number of samples for quick testing
            "num_pert_samples": 5,
            "perturbed_noise": 0.01,
            "data_dir": ".",            # Current directory
            "device": "cpu",            # Use CPU for local testing
            "outdir": "test_run"
        }
    else:
        # Fuller configuration for cluster testing
        config = {
            "K": 100,
            "epochs": 100,              # More epochs but still not full training
            "step_size": 0.001,
            "bpr_weight": 1.0,
            "nll_weight": 1.0,
            "seed": 42,
            "threshold": 0.55,
            "num_score_samples": 20,
            "num_pert_samples": 20,
            "perturbed_noise": 0.01,
            "data_dir": "/cluster/home/kheuto01/code/prob_diff_topk",
            "device": "cuda",
            "outdir": "cluster_test_run"
        }
    
    # Convert config to command line arguments
    args = " ".join([f"--{k} {v}" for k, v in config.items()])
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Generate local testing configuration")
    parser.add_argument("--print_only", action="store_true", help="Only print the arguments, don't execute")
    args = parser.parse_args()
    
    # Get the test configuration
    cmd_args = get_test_config(local=args.local)
    
    if args.print_only:
        print(cmd_args)
    else:
        # Execute the script with the test configuration
        if args.local:
            os.system(f"python negative_binomial_training.py {cmd_args}")
        else:
            code_dir = "/cluster/home/kheuto01/code/prob_diff_topk"
            os.system(f"code_dir={code_dir} args='{cmd_args}' sbatch < /cluster/home/kheuto01/code/prob_diff_topk/run_neg_binom.slurm")