import torch

import torch.nn as nn
from torch.distributions import Categorical, Poisson, MixtureSameFamily, NegativeBinomial
from torch_distributions import TruncatedNormal


class NegativeBinomialRegressionModel(nn.Module):
    def __init__(self, num_locations, num_fixed_effects, low=0, high=float('inf'), device='cpu'):
        super(NegativeBinomialRegressionModel, self).__init__()
        self.num_locations = num_locations
        self.num_fixed_effects = num_fixed_effects
        self.low = low
        self.high = high

        # Fixed effects
        self.beta_0 = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(num_fixed_effects))

        # Random effects
        self.b_0 = torch.zeros(num_locations).to(device)
        self.b_1 = torch.zeros(num_locations).to(device)

        # Covariance matrix parameters
        self.log_sigma_0 = nn.Parameter(torch.randn(1))
        self.log_sigma_1 = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.randn(1))

        # probability param
        self.softinv_theta = nn.Parameter(torch.randn(1))

    def forward(self, X, time):
        # Ensure X and time have the correct shape
        assert X.shape[1] == self.num_locations, "X should have shape (num_time_points, num_locations, num_fixed_effects)"
        assert X.shape[2] == self.num_fixed_effects, "X should have shape (num_time_points, num_locations, num_fixed_effects)"
        assert time.shape == X.shape[:2], "time should have shape (num_time_points, num_locations)"

        # Calculate mu
        fixed_effects = self.beta_0 + torch.einsum('tli,i->tl', X, self.beta)
        random_intercepts = self.b_0.expand(X.shape[0], -1)
        random_slopes = self.b_1.expand(X.shape[0], -1)
        
        log_mu = fixed_effects + random_intercepts + random_slopes * time

        # Use softplus to ensure mu is positive and grows more slowly
        mu = nn.functional.softplus(log_mu)

        # Calculate theta probability
        theta = torch.nn.functional.softplus(self.softinv_theta)

        # Calculate logits instead of probabilities
        logits = torch.log(mu) - torch.log(theta)
        print(f'unconstrained theta: {self.softinv_theta}')
        print(f'Theta: {theta}')

        # Create and return the NegativeBinomial distribution using logits
        return NegativeBinomial(total_count=mu, probs=theta)
    def get_covariance_matrix(self):
        sigma_0 = torch.exp(self.log_sigma_0)
        sigma_1 = torch.exp(self.log_sigma_1)
        rho = torch.tanh(self.rho)  # Ensure -1 < rho < 1
        
        # Construct the covariance matrix using operations that maintain the computational graph
        cov_00 = sigma_0 * sigma_0
        cov_01 = rho * sigma_0 * sigma_1
        cov_11 = sigma_1 * sigma_1
        
        cov_matrix = torch.stack([
            torch.stack([cov_00, cov_01]),
            torch.stack([cov_01, cov_11])
        ])

        cov_matrix = torch.squeeze(cov_matrix)
        
        return cov_matrix

    def log_likelihood(self, y, X, time):
        dist = self.forward(X, time)
        log_prob = dist.log_prob(y)
        
        # Add penalty for random effects
        cov_matrix = self.get_covariance_matrix()
        random_effects = torch.stack([self.b_0, self.b_1], dim=1)
        penalty = -0.5 * torch.mean(
            torch.matmul(random_effects, torch.inverse(cov_matrix)) * random_effects
        )
        
        return torch.mean(log_prob) + penalty

    def sample(self, X, time, num_samples=1):
        dist = self.forward(X, time)
        samples = dist.sample((num_samples,))
        return torch.clamp(samples, self.low, self.high)

    def params_to_single_tensor(self):
        return torch.cat([param.view(-1) for param in self.parameters()])

    def single_tensor_to_params(self, single_tensor):
        params = []
        idx = 0
        for param in self.parameters():
            num_elements = param.numel()
            param_data = single_tensor[idx:idx+num_elements].view(param.shape)
            params.append(param_data)
            idx += num_elements
        return params

    def update_params(self, single_tensor):
        params = self.single_tensor_to_params(single_tensor)
        for param, new_data in zip(self.parameters(), params):
            param.data = new_data

    def build_from_single_tensor(self, single_tensor, X, time):
        self.update_params(single_tensor)
        return self.forward(X, time)

    def plot_learned(self, X, time, data=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        dist = self.forward(X, time)
        samples = dist.sample((1000,)).view(-1).detach().numpy()
        
        if ax is None:
            fig, ax = plt.subplots()
        
        sns.kdeplot(samples, bw_adjust=0.5, ax=ax)
        
        if data is not None:
            sns.histplot(data.flatten(), stat='density', bins=50, color='red', alpha=0.5, ax=ax)
        
        plt.show()
        
        return ax

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

class SpatialWaves(nn.Module):

    def __init__(self, num_waves=2, min_peak_width=0.1, death_noise=0.2, low=0, high=100, has_speed=False):
        super(SpatialWaves, self).__init__()
        self.low = low
        self.high = high
        self.num_waves = num_waves
        self.has_speed = has_speed
        if self.has_speed:
            self.arrival_speeds =nn.Parameter(torch.rand(num_waves), requires_grad=True)
        self.arrival_intercepts =nn.Parameter(torch.rand(num_waves)+12, requires_grad=True)
        self.lat_coeff = nn.Parameter(-torch.rand(num_waves), requires_grad=True)
        self.lon_coeff = nn.Parameter(-torch.rand(num_waves), requires_grad=True)
        # magnitudes are random between -6 and -5
        self.softplusinv_magnitudes = nn.Parameter(torch.rand(num_waves)-6.0, requires_grad=True)
        self.softplusinv_peak_widths = nn.Parameter(torch.rand(num_waves)+2.0, requires_grad=True)
        self.min_peak_width=min_peak_width
        self.death_noise = death_noise

    def params_to_single_tensor(self):
        return torch.cat([param.view(-1) for param in self.parameters()])

    def single_tensor_to_params(self, single_tensor):
        if self.has_speed:
            arrival_speeds = single_tensor[:self.num_waves]
            speed_offset = 1
        else:
            speed_offset = 0
        arrival_intercepts = single_tensor[self.num_waves*(speed_offset):(1+speed_offset)*self.num_waves]
        lat_coeff = single_tensor[(1+speed_offset)*self.num_waves:(2+speed_offset)*self.num_waves]
        lon_coeff = single_tensor[(2+speed_offset)*self.num_waves:(3+speed_offset)*self.num_waves]
        softplusinv_magnitudes = single_tensor[(3+speed_offset)*self.num_waves:(4+speed_offset)*self.num_waves]
        softplusinv_peak_widths = single_tensor[(4+speed_offset)*self.num_waves:(5+speed_offset)*self.num_waves]
        if self.has_speed:
            return arrival_speeds, arrival_intercepts, lat_coeff, lon_coeff, softplusinv_magnitudes, softplusinv_peak_widths
        else:
            return arrival_intercepts, lat_coeff, lon_coeff, softplusinv_magnitudes, softplusinv_peak_widths
    
    def update_params(self, single_tensor):
        if self.has_speed:
            arrival_speeds, arrival_intercepts, lat_coeff, lon_coeff, softplusinv_magnitudes, softplusinv_peak_widths = self.single_tensor_to_params(single_tensor)
            self.arrival_speeds.data = arrival_speeds
        else:
            arrival_intercepts, lat_coeff, lon_coeff, softplusinv_magnitudes, softplusinv_peak_widths = self.single_tensor_to_params(single_tensor)
        self.arrival_intercepts.data = arrival_intercepts
        self.lat_coeff.data = lat_coeff
        self.lon_coeff.data = lon_coeff
        self.softplusinv_magnitudes.data = softplusinv_magnitudes
        self.softplusinv_peak_widths.data = softplusinv_peak_widths

        return
    
    def build_from_single_tensor(self, single_tensor, time_T, pop_S, lat_S, lon_S):

        S = pop_S.shape[0]
        T = time_T.shape[0]
        W = self.num_waves
        if self.has_speed:
            arrival_speeds, arrival_intercepts, lat_coeff, lon_coeff, softplusinv_magnitudes, softplusinv_peak_widths = self.single_tensor_to_params(single_tensor)
        else:
            arrival_intercepts, lat_coeff, lon_coeff, softplusinv_magnitudes, softplusinv_peak_widths = self.single_tensor_to_params(single_tensor)
        
        magnitudes_W = torch.nn.functional.softplus(softplusinv_magnitudes)
        peak_widths_W = torch.nn.functional.softplus(softplusinv_peak_widths) + self.min_peak_width
        arrival_times_SW = arrival_intercepts.expand(S,W) +lat_S.unsqueeze(-1)*lat_coeff.expand(1,W) + lon_S.unsqueeze(-1)*lon_coeff.expand(1,W)
        
        # exponentiate time - arrival time / peak width
        if self.has_speed:
            time_diff_TSW = arrival_speeds.expand(1,1,W)*time_T.unsqueeze(-1).unsqueeze(-1) - arrival_times_SW.expand(1,S,W)
        else:
            time_diff_TSW = time_T.unsqueeze(-1).unsqueeze(-1) - arrival_times_SW.expand(1,S,W)
        # square time_diff/peak_width
        waves_TSW = magnitudes_W.expand(1,1,W)*torch.exp(-(time_diff_TSW/peak_widths_W.expand(1,1,W))**2)
        death_rate_TS = torch.sum(waves_TSW, dim=-1)
        mean_deaths_TS = pop_S.expand(1,S)*death_rate_TS
        return TruncatedNormal(mean_deaths_TS, self.death_noise, self.low, self.high, validate_args=False)
    
    def forward(self, time_T, pop_S, lat_S, lon_S):

        S = pop_S.shape[0]
        T = time_T.shape[0]
        W = self.num_waves

        magnitudes_W = torch.nn.functional.softplus(self.softplusinv_magnitudes)
        peak_widths_W = torch.nn.functional.softplus(self.softplusinv_peak_widths) + self.min_peak_width


        arrival_times_SW = self.arrival_intercepts.expand(S,W) +lat_S.unsqueeze(-1)*self.lat_coeff.expand(1,W) + lon_S.unsqueeze(-1)*self.lon_coeff.expand(1,W)
        
        # exponentiate time - arrival time / peak width
        if self.has_speed:
            time_diff_TSW = self.arrival_speeds.expand(1,1,W)*time_T.unsqueeze(-1).unsqueeze(-1) - arrival_times_SW.expand(1,S,W)
        else:
            time_diff_TSW = time_T.unsqueeze(-1).unsqueeze(-1) - arrival_times_SW.expand(1,S,W)
            print(time_diff_TSW.shape)
        # square time_diff/peak_width
        waves_TSW = magnitudes_W.expand(1,1,W)*torch.exp(-(time_diff_TSW/peak_widths_W.expand(1,1,W))**2)
        death_rate_TS = torch.sum(waves_TSW, dim=-1)
        mean_deaths_TS = pop_S.expand(1,S)*death_rate_TS
        return TruncatedNormal(mean_deaths_TS, self.death_noise, self.low, self.high, validate_args=False)

    def plot_learned(self,time, pop, lat, lon, data=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        dist = self.forward(time, pop, lat, lon)

        samples = dist.sample((1000,))
        samples = samples.view(-1).detach().numpy()
        # plot the kde only
        sns.kdeplot(samples, bw_adjust=0.5, ax=ax)
        # plot hist of data if present
        if data is not None:
            sns.histplot(data.flatten(), stat='density', bins=50, color='red', alpha=0.5, ax=ax)
        plt.show()
        
        return

    
class MixtureOfTruncNormModel(nn.Module):
    def __init__(self, num_components=4, S=12, low=0, high=200):
        super(MixtureOfTruncNormModel, self).__init__()
        self.num_components = num_components
        self.S = S
        self.low=low
        self.high=high
        
        # Initialize the log rates and mixture probabilities as learnable parameters
        self.softplusinv_means = nn.Parameter(torch.rand(num_components)*50, requires_grad=True) 
        self.softplusinv_scales = nn.Parameter(torch.rand(num_components), requires_grad=True) 
        self.mixture_probs = nn.Parameter(torch.rand(S, num_components), requires_grad=True)  

    def params_to_single_tensor(self):
        return torch.cat([param.view(-1) for param in self.parameters()])
    
    def single_tensor_to_params(self, single_tensor):
        softplusinv_means = single_tensor[:self.num_components]
        softplusinv_scales = single_tensor[self.num_components:2*self.num_components]
        mixture_probs = single_tensor[2*self.num_components:].view(self.S, self.num_components)
        return softplusinv_means, softplusinv_scales, mixture_probs
    
    def update_params(self, single_tensor):
        softplusinv_means, softplusinv_scales, mixture_probs = self.single_tensor_to_params(single_tensor)
        self.softplusinv_means.data = softplusinv_means
        self.softplusinv_scales.data = softplusinv_scales
        self.mixture_probs.data = mixture_probs
        return
    
    def build_from_single_tensor(self, single_tensor):
        softplusinv_means, softplusinv_scales, mixture_probs = self.single_tensor_to_params(single_tensor)
        means = torch.nn.functional.softplus(softplusinv_means)
        scales = torch.nn.functional.softplus(softplusinv_scales) + 0.2
        mixture_probs_normalized = torch.nn.functional.softmax(mixture_probs, dim=1)
        categorical_dist = Categorical(mixture_probs_normalized)
        expanded_means = means.expand(self.S, self.num_components)
        expanded_scales = scales.expand(self.S, self.num_components)
        trunc_norm_dist = TruncatedNormal(expanded_means, expanded_scales, self.low, self.high, validate_args=False)
        mixture_dist = MixtureSameFamily(categorical_dist, trunc_norm_dist, validate_args=False)
        return mixture_dist
        
    def forward(self):
        means = torch.nn.functional.softplus(self.softplusinv_means)
        scales = torch.nn.functional.softplus(self.softplusinv_scales) + 0.2
        mixture_probs_normalized = torch.nn.functional.softmax(self.mixture_probs, dim=1)
        categorical_dist = Categorical(mixture_probs_normalized)
        expanded_means = means.expand(self.S, self.num_components)
        expanded_scales = scales.expand(self.S, self.num_components)
        trunc_norm_dist = TruncatedNormal(expanded_means, expanded_scales, self.low, self.high)
        
        # Create the MixtureSameFamily distribution
        mixture_dist = MixtureSameFamily(categorical_dist, trunc_norm_dist,  validate_args=False)
        
        
        
        return mixture_dist
    
    def plot_learned(self, data=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        mixture_dist = self.forward()

        samples = mixture_dist.sample((1000,))
        samples = samples.view(-1).detach().numpy()
        # plot the kde only
        sns.kdeplot(samples, bw_adjust=0.5, ax=ax)
        # plot hist of data if present
        if data is not None:
            sns.histplot(data.flatten(), stat='density', bins=50, color='red', alpha=0.5, ax=ax)
        plt.show()
        
        return

    
def torch_bpr_uncurried(y_pred, y_true, K=4, perturbed_top_K_func=None):

    top_K_ids = perturbed_top_K_func(y_pred)
    # Sum over k dim
    top_K_ids = top_K_ids.sum(dim=-2)

    true_top_K_val, _  = torch.topk(y_true, K) 
    denominator = torch.sum(true_top_K_val, dim=-1)
    numerator = torch.sum(top_K_ids * y_true, dim=-1)
    bpr = numerator/denominator

    return bpr

def deterministic_bpr(y_pred, y_true, K=4):

    true_topk = torch.topk(y_true, K)
    pred_topk = torch.topk(y_pred, K)
    # index y_pred with top k indicies from y_true
    true_value_at_pred_topk = torch.gather(y_true, 1, pred_topk.indices)

    numerator = torch.sum(true_value_at_pred_topk, dim=-1)
    denominator = torch.sum(true_topk.values, dim=-1)
    bpr = numerator/denominator
    return bpr
