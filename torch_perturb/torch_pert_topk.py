import torch
from torch import nn


class PerturbedBrokenTopK(nn.Module):

    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedBrokenTopK, self).__init__()
    
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):
        # Return the output of the PerturbedTopKFunction, applied to the input tensor
        # using the k, num_samples, and sigma attributes as arguments
        return PerturbedBrokenTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedBrokenTopKFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
    
        b, d = x.shape
        
        # Generate Gaussian noise with specified number of samples and standard deviation
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        # Add noise to the input tensor
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        
        # Perform top-k pooling on the perturbed tensor
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        
        # Get the indices of the top k elements
        indices = topk_results.indices # b, nS, k
        
        # Sort the indices in ascending order
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        # Convert the indices to one-hot tensors
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        
        # Average the one-hot tensors to get the final output
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # Save constants and tensors for backward pass
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators


    @staticmethod
    def backward(ctx, grad_output):
        # If there is no gradient to backpropagate, return tuple of None values
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        
        # Calculate expected gradient
        expected_gradient = (
            torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        ) * float(ctx.k)
        
        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)
        
        return (grad_input,) + tuple([None] * 5)

