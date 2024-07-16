import os
import torch

from datasets import example_datasets, to_numpy
from torch_models import  MixtureOfTruncNormModel, torch_bpr_uncurried, deterministic_bpr
from torch_perturb.torch_pert_topk import PerturbedTopK
from torch_training import training_loop

def main(step_size=None, epochs=None, bpr_weight=None, nll_weight=None, seed=None, outdir=None, threshold=None):

    # tracts/distributions
    S=12
    # history/features
    H = 3
    # total timepoints
    T= 500
    K=4

    train_dataset, val_dataset, test_dataset = example_datasets(H, T, seed=seed)
    train_X_THS, train_y_TS = to_numpy(train_dataset)

    model = MixtureOfTruncNormModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    M_score_func =  200
    M_action = 200
    train_T = train_y_TS.shape[0]
    perturbed_top_K_func = PerturbedTopK(k=K)

    losses, bprs, nlls, model = training_loop(epochs, model, optimizer, K, threshold, train_T, M_score_func, M_action, train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight)
    
    # save everything to outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    torch.save(model.state_dict(), f'{outdir}/model.pth')
    torch.save(optimizer.state_dict(), f'{outdir}/optimizer.pth')
    torch.save(losses, f'{outdir}/losses.pth')
    torch.save(bprs, f'{outdir}/bprs.pth')
    torch.save(nlls, f'{outdir}/nlls.pth')




if __name__ ==  "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bpr_weight", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)

    args = parser.parse_args()
    # call main with args as keywrods
    main(**vars(args))