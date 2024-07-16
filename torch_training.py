import torch
import sys
from torch_models import torch_bpr_uncurried, deterministic_bpr
def training_loop(epochs, model, optimizer, K, threshold, train_T, M_score_func, M_action, train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight):
    losses = []
    bprs = []
    nlls = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        mix_model = model()
        
        y_sample_TMS = mix_model.sample((train_T, M_score_func))
        y_sample_action_TMS = mix_model.sample((train_T, M_action))

        ratio_rating_TMS = y_sample_action_TMS/y_sample_action_TMS.sum(dim=-1, keepdim=True)
        ratio_rating_TS =  ratio_rating_TMS.mean(dim=1)
        ratio_rating_TS.requires_grad_(True)

        def get_log_probs_baked(param):
            distribution = model.build_from_single_tensor(param)
            log_probs_TMS = distribution.log_prob(y_sample_TMS)

            return log_probs_TMS
        
        jac_TMSP = torch.autograd.functional.jacobian(get_log_probs_baked, (model.params_to_single_tensor()), strategy='forward-mode', vectorize=True)

        score_func_estimator_TMSP = jac_TMSP * ratio_rating_TMS.unsqueeze(-1)
        score_func_estimator_TSP = score_func_estimator_TMSP.mean(dim=1)    

        # get gradient of negative bpr_t  with respect to ratio rating_TS
        positive_bpr_T = torch_bpr_uncurried(ratio_rating_TS, torch.tensor(train_y_TS), K=K, perturbed_top_K_func=perturbed_top_K_func)
        bpr_threshold_diff_T = positive_bpr_T - threshold
        violate_threshold_flag = bpr_threshold_diff_T < 0
        negative_bpr_loss = torch.mean(-bpr_threshold_diff_T*violate_threshold_flag)
        
        nll = torch.sum(-mix_model.log_prob( torch.tensor(train_y_TS)))

        loss = bpr_weight*negative_bpr_loss + nll_weight*nll
        loss.backward()
        loss_grad_TS = ratio_rating_TS.grad

        gradient_TSP = score_func_estimator_TSP * torch.unsqueeze(loss_grad_TS, -1)
        gradient_P = torch.sum(gradient_TSP, dim=[0,1])

        gradient_tuple = model.single_tensor_to_params(gradient_P)

        for param, gradient in zip(model.parameters(), gradient_tuple):
            if nll_weight>0:
                gradient = gradient + param.grad
            param.grad = gradient
        optimizer.step()

        deterministic_bpr_T = deterministic_bpr(ratio_rating_TS, torch.tensor(train_y_TS), K=K)
        det_bpr =torch.mean(deterministic_bpr_T)
        print(f'Epoch: {epoch}')
        print(f'det bpr: {det_bpr}')
        print(f'nll: {nll}')
        print(f'Loss: {loss}')
        losses.append(loss)
        bprs.append(det_bpr)
        nlls.append(nll)
        sys.stdout.flush()

    return losses, bprs, nlls, model