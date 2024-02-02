import tensorflow as tf
from perturbations.perturbations import perturbed
from functools import partial


def poisson_nll(y_true, y_pred):
    """Poisson negative log likelihood loss function.
    
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        negative log likelihood
    """
    return y_pred - y_true * tf.math.log(y_pred + 1e-10)

def top_k_idx(input_BD, **kwargs):
    _, idx_BD = tf.math.top_k(input_BD, **kwargs)
    input_depth = input_BD.shape[-1]
    one_hot_idx_BKD = tf.one_hot(idx_BD, input_depth)
    # Sum over k dimension so we dont have to worry about sorting
    k_hot_idx_BD = tf.reduce_sum(one_hot_idx_BKD, axis=-2)

    return k_hot_idx_BD

def negative_bpr_K_uncurried(y_true, y_pred, K=3, perturbed_top_K_func=None):
    
    top_K_ids = perturbed_top_K_func(y_pred)
    true_top_K_val, true_top_K_idx = tf.math.top_k(y_true, k=K)
    denominator = tf.reduce_sum(true_top_K_val, axis=-1)
    numerator = tf.reduce_sum(top_K_ids * y_true, axis=-1)
    loss_val = tf.reduce_mean(-numerator/denominator)

    return loss_val

def get_bpr_loss_func(K, num_samples=1000, sigma=1, noise='normal'):

    top_K_idx_func = partial(top_k_idx, k=K)
    perturbed_top_K = perturbed(top_K_idx_func,
                                    num_samples=num_samples,
                                    sigma=sigma,
                                    noise=noise,
                                    batched=True)

    negative_bpr_K = partial(negative_bpr_K_uncurried,K=K, perturbed_top_K_func=perturbed_top_K)

    return negative_bpr_K

def uncurried_penalized_bpr(y_true, y_pred, penalty=1.0, 
                            bpr_threshold=0.40, negative_bpr_K_func=None):
    
    negative_bpr_K_val = negative_bpr_K_func(y_true, y_pred)
    # threshold is violated when threshold is above bpr
    violate_threshold_flag = tf.cast(tf.greater(bpr_threshold,
                                                -negative_bpr_K_val),
                                     tf.float32)
    
    loss = poisson_nll(y_true, y_pred) + penalty * violate_threshold_flag *(bpr_threshold + negative_bpr_K_val)

    return loss

def get_penalized_bpr_loss_func(K, penalty, bpr_threshold,
                                num_samples=1000, sigma=1, noise='normal'):
    negative_bpr_K_func = get_bpr_loss_func(K, num_samples, sigma, noise)
    penalized_bpr = partial(uncurried_penalized_bpr, penalty=penalty, 
                            bpr_threshold=bpr_threshold, 
                            negative_bpr_K_func=negative_bpr_K_func)

    return penalized_bpr