import os
import torch
import numpy as np

from distributions import QuantizedNormal
from torch_models import  MixtureOfTruncNormModel, torch_bpr_uncurried, deterministic_bpr
from torch_perturb.torch_pert_topk import PerturbedTopK
from torch_training import train_epoch

def main(step_size=None, epochs=None, bpr_weight=None,
         nll_weight=None, seed=None, outdir=None, threshold=None,
           num_components=None, perturbed_noise=None, initialization=None,
           mu1=None, mu2=None):

    # tracts/distributions
    S=23

    # total timepoints
    T= 500
    K=3

    low_set_1_10 = [QuantizedNormal(10, 0.3) for _ in range(10)]
    low_set_2_10 = [QuantizedNormal(30, 0.3) for _ in range(10)]
    high_set_3 = [QuantizedNormal(50,3) for _ in range(3)]

    dist_S = low_set_1_10 + low_set_2_10 + high_set_3

    train_y_TS = np.zeros((T, S))
    for s, dist in enumerate(dist_S):
        random_state = np.random.RandomState(10000 * seed + s*123456)
        train_y_TS[:, s] = dist.rvs(size=T, random_state=random_state)

    model = MixtureOfTruncNormModel(num_components=num_components, S=S, low=0, high=100)
    if initialization == 'bpr':
        means = torch.tensor([1, 10])
        softinv_means = means + torch.log(-torch.expm1(-means))
        scales = torch.tensor([0.25, 0.25])
        softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2)) 
        mix_weights = torch.log(1e-13 + torch.tensor(
                                        [[1,0]*20+[0,1]*3]))
        model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))
    elif initialization == 'nll':
        means = torch.tensor([10, 37])
        softinv_means = means + torch.log(-torch.expm1(-means))
        scales = torch.tensor([0.3, 5])
        softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2)) 
        mix_weights = torch.log(1e-13 + torch.tensor(
                                        [[1,0]*10+[0,1]*13]))
        model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))

    if mu1 == 10:
        if mu2 == 30:
            means = torch.tensor([10, 30])
            softinv_means = means + torch.log(-torch.expm1(-means))
            scales = torch.tensor([1, 1])
            softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2)) 
            mix_weights = torch.log(1e-13 + torch.tensor(
                                            [[0.99,0.1]*10+[0.1,0.99]*13]))
            model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))
        elif mu2 == 50:
            means = torch.tensor([10, 50])
            softinv_means = means + torch.log(-torch.expm1(-means))
            scales = torch.tensor([1, 1])
            softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2)) 
            mix_weights = torch.log(1e-13 + torch.tensor(
                                            [[0.99,0.1]*10 + [0.5, 0.5]*10 + [0.1,0.99]*3]))
            model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))
    elif mu1 == 30:
        means = torch.tensor([30, 50])
        softinv_means = means + torch.log(-torch.expm1(-means))
        scales = torch.tensor([1, 1])
        softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2))
        mix_weights = torch.log(1e-13 + torch.tensor(
                                        [[0.99,0.1]*20+[0.1,0.99]*3]))


    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    M_score_func =  100
    M_action = 100
    train_T = train_y_TS.shape[0]
    perturbed_top_K_func = PerturbedTopK(k=K, sigma=perturbed_noise)
    losses, bprs, nlls = [], [], []
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        loss, bpr, nll, model = train_epoch(model, optimizer, K, threshold, train_T, M_score_func, M_action, train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight)
        losses.append(loss)
        bprs.append(bpr)
        nlls.append(nll)
    
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
    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument("--perturbed_noise", type=float, default=0.05)
    parser.add_argument("--initialization", type=str, required=False)
    parser.add_argument("--mu1", type=int, required=False)
    parser.add_argument("--mu2", type=int, required=False)

    args = parser.parse_args()
    # call main with args as keywrods
    main(**vars(args))