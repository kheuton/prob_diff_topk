import os
import torch
import numpy as np
import pandas as pd
import time
from functools import partial
from distributions import QuantizedNormal
from torch_models import  MixtureOfTruncNormModel, torch_bpr_uncurried, deterministic_bpr, SpatialWaves
#from torch_perturb.torch_pert_topk import PerturbedTopK
from metrics import top_k_onehot_indicator
from torch_training import train_epoch_largesynth
from torch_perturb.perturbations import perturbed

def main(K=None,step_size=None, epochs=None, bpr_weight=None,
         nll_weight=None, seed=None, init_idx=None, outdir=None, threshold=None,
         perturbed_noise=None, initialization=None, num_score_samples=None, num_pert_samples=None,
         data_dir='/cluster/home/kheuto01/code/prob_diff_topk'):


    rows=27
    cols=60
    data_shape=(rows, cols)
    deaths = pd.read_csv(os.path.join(data_dir,'deaths_bandheavy.csv'))
    pop = pd.read_csv(os.path.join(data_dir, 'pop_bandheavy.csv'))

    # turn the death column into a time-by-geoid array
    deaths_TS = deaths.pivot(index='time', columns='geoid', values='death').values
    pop_S = pop['pop'].values

    T, S = deaths_TS.shape
    # create latitude and longitude arrays corresponding to the row and column index of the geoids when reshaped into data_shape\
    lat = np.linspace(-rows/2, rows/2, rows)
    lon = np.linspace(-cols/2, cols/2, cols)
    lat_S, lon_S = np.meshgrid(lon, lat)
    lat_S = lat_S.flatten()
    lon_S = lon_S.flatten()
    # create column of time values
    time_T = np.arange(deaths_TS.shape[0])
    time_T = torch.tensor(time_T, dtype=torch.float32)
    lat_S = torch.tensor(lat_S, dtype=torch.float32)
    lon_S = torch.tensor(lon_S, dtype=torch.float32)
    pop_S = torch.tensor(pop_S, dtype=torch.float32)
    S = pop_S.shape[0]
    T = time_T.shape[0]


    model  = SpatialWaves(num_waves=1,low=0, high=1000000)

    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    M_score_func =  num_score_samples
    M_action = M_score_func

    top_k_func = partial(top_k_onehot_indicator, k=K)
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=num_pert_samples)
    losses, bprs, nlls, times = [], [], [], []
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        start = time.time()
        loss, bpr, nll, model = train_epoch_largesynth(model, optimizer, K, threshold, T,
                                            M_score_func, M_action,time_T,pop_S,
                                            lat_S, lon_S, deaths_TS, 
                                            perturbed_top_K_func, bpr_weight, 
                                            nll_weight)
        end = time.time()
        elapsed = end - start
        losses.append(loss)
        bprs.append(bpr)
        nlls.append(nll)
        times.append(elapsed)
    
        # save everything to outdir every 100 epochs
        if epoch % 100 == 0:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            torch.save(model.state_dict(), f'{outdir}/model.pth')
            torch.save(optimizer.state_dict(), f'{outdir}/optimizer.pth')
            torch.save(losses, f'{outdir}/losses.pth')
            torch.save(bprs, f'{outdir}/bprs.pth')
            torch.save(nlls, f'{outdir}/nlls.pth')
            torch.save(times, f'{outdir}/times.pth')




if __name__ ==  "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bpr_weight", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--perturbed_noise", type=float, default=0.05)
    parser.add_argument("--initialization", type=str, required=False)
    parser.add_argument("--num_score_samples", type=int, required=False, default=50)
    parser.add_argument("--num_pert_samples", type=int, required=False, default=50)
    parser.add_argument("--data_dir", type=str, default='/cluster/home/kheuto01/code/prob_diff_topk')

    args = parser.parse_args()
    # call main with args as keywrods
    main(**vars(args))