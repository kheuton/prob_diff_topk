import torch

import torch.nn as nn
from torch.distributions import Categorical, Poisson, MixtureSameFamily


class MixtureOfPoissonsModel(nn.Module):
    def __init__(self, num_components=4, S=12):
        super(MixtureOfPoissonsModel, self).__init__()
        self.num_components = num_components
        self.S = S
        
        # Initialize the log rates and mixture probabilities as learnable parameters
        self.log_poisson_rates = nn.Parameter(torch.rand(num_components), requires_grad=True)  # Initialize log rates
        self.mixture_probs = nn.Parameter(torch.rand(S, num_components), requires_grad=True)  # Initialize probabilities

    def params_to_single_tensor(self):
        return torch.cat([self.log_poisson_rates, self.mixture_probs.view(-1)])
    
    def single_tensor_to_params(self, single_tensor):
        log_poisson_rates = single_tensor[:self.num_components]
        mixture_probs = single_tensor[self.num_components:].view(self.S, self.num_components)
        return log_poisson_rates, mixture_probs
    
    def update_params(self, single_tensor):
        log_poisson_rates, mixture_probs = self.single_tensor_to_params(single_tensor)
        self.log_poisson_rates.data = log_poisson_rates
        self.mixture_probs.data = mixture_probs
        return
    
    def build_from_single_tensor(self, single_tensor):
        log_poisson_rates, mixture_probs = self.single_tensor_to_params(single_tensor)
        poisson_rates = torch.exp(log_poisson_rates)
        mixture_probs_normalized = torch.nn.functional.softmax(mixture_probs, dim=1)
        categorical_dist = Categorical(mixture_probs_normalized)
        expanded_rates = poisson_rates.expand(self.S, self.num_components)
        poisson_dist = Poisson(expanded_rates, validate_args=False)
        mixture_dist = MixtureSameFamily(categorical_dist, poisson_dist)
        return mixture_dist
        
    def forward(self):
        # Transform log rates to rates
        poisson_rates = torch.exp(self.log_poisson_rates)
        
        # Normalize mixture_probs to sum to 1 across the components
        mixture_probs_normalized = torch.nn.functional.softmax(self.mixture_probs, dim=1)
        
        # Create the Categorical distribution with the normalized probabilities
        categorical_dist = Categorical(mixture_probs_normalized)
        
        # Expand the Poisson rates to match the number of samples
        expanded_rates = poisson_rates.expand(self.S, self.num_components)
        
        # Create the Poisson distribution with the expanded rates
        poisson_dist = Poisson(expanded_rates, validate_args=False)
        
        # Create the MixtureSameFamily distribution
        mixture_dist = MixtureSameFamily(categorical_dist, poisson_dist,  validate_args=False)
        
        
        
        return mixture_dist
    
def torch_bpr_uncurried(y_pred, y_true, K=4, perturbed_top_K_func=None):

    top_K_ids = perturbed_top_K_func(y_pred)
    # Sum over k dim
    top_K_ids = top_K_ids.sum(dim=-2)

    true_top_K_val, _  = torch.topk(y_true, K) 
    denominator = torch.sum(true_top_K_val, dim=-1)
    numerator = torch.sum(top_K_ids * y_true, dim=-1)
    bpr = numerator/denominator

    return bpr