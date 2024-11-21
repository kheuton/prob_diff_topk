# Raw data to outputs

1. Clean data with opioid repo 
2. Run GetRealDataExps.ipynb
    - TODO: replace with script
    - TODO: add covariates here 
3. Edit torch_opioid_exp.py to be your model/datasets
    - TODO: this could be better
4. Enter slurm config into run_neg_binom.slurm
4. Enter hyperparameter grid into launch_mass_exps.py
    - TODO: this could be a separate config/ more options for models and datasets
5. Call it, look at results
