# Decision Aware Maximum Likelihood
This anonymous repository contains the code for the paper "Decision-aware training of spatiotemporal forecasting models to select a top K subset of sites for intervention," currently in submission

## Synthetic Data Experiment
To replicate the results of the synthetic data experiment, run the following command:
```
python torch_frontier_experiment.py --step_size 0.1 \
    --perturbed_noise 0.01 \
    --epochs 2000 \
    --bpr_weight 30 --nll_weight 1 \
    --seed 360 \
    --outdir ./output/  \
    --num_components 2 \
    --init_idx 0 \
    --epsilon 0.84
```

Try adjusting epsilon to encourage the model to achieve different BPRs. The model will reach the target BPRs quickly, but the minimum NLL requires several thousand epochs.

## Opioid-related Overdose Experiment
We share the publically available Cook County Dataset, preprocessed to fit our modeling strategy. Our experiments can be run with commands like the following:

```
python torch_opioid_exp.py --step_size 0.1 \
    --perturbed_noise 0.01 \
    --epochs 20 \
    --bpr_weight 30 \
    --nll_weight 1 \
    --seed 360 
    --data_dir ./data/cook_county/ \
    --device cuda \
    --outdir ./output/ \
    --epsilon 1
```
