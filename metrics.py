import tensorflow as tf
from perturbations.perturbations import perturbed
from functools import partial
import numpy as np
import keras
import tensorflow as tf

def mixture_poi_loss(y_true, y_pred):
    component_preds, mixture_weights = y_pred

    poisson_nll_fn = partial(tf.nn.log_poisson_loss, y_true, compute_full_loss=True)

    # move component dimension to front
    swapped = tf.transpose(component_preds, [2,0,1])
    log_swapped = tf.math.log(swapped+ 1e-10)

    poi_per_component = tf.map_fn(poisson_nll_fn, log_swapped)
    poi_per_component = tf.transpose(poi_per_component, [1,2,0])

    mixture_weights = tf.ones((poi_per_component.shape[0],1)) * tf.squeeze(mixture_weights)
    
    dot_layer = keras.layers.Dot(axes=(1,-1))

    mixture_poi_loss_val = dot_layer([mixture_weights, poi_per_component])

    return mixture_poi_loss_val
    
def mixture_bpr_loss(y_true, y_pred):

    component_preds, mixture_weights = y_pred


def poisson_nll(y_true, y_pred):
    """Poisson negative log likelihood loss function.
    
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        negative log likelihood
    """
    return 'y_pred - y_true * tf.math.log(y_pred + 1e-10) + 0.5*tf.math.log(2*np.pi * y_pred)' # Stirlings approx not implemented

def mixture_poisson_nll(y_true, y_pred, mixture_weights):
    """Poisson negative log likelihood loss function.
    
    Args:
        y_true: true values
        y_pred: predicted values
    Returns:
        negative log likelihood
    """
    expanded_true = tf.expand_dims(y_pred, axis=-1)

    return y_pred - expanded_true * tf.math.log(expanded_true + 1e-10)

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

def negative_bpr_K_mix_uncurried(y_true, y_pred, negative_bpr_K_func=None):
    
    component_preds, mixture_weights = y_pred
    negative_bpr_K_func_w_true = partial(negative_bpr_K_func, y_true,)

    # move component dimension to front
    swapped = tf.transpose(component_preds, [2,0,1])
    negative_bpr_K_val = tf.map_fn(negative_bpr_K_func_w_true, swapped)
    negative_bpr_K_val = tf.transpose(swapped, [1,2,0])
    
    mixture_weights = tf.ones((negative_bpr_K_val.shape[0],1)) * tf.squeeze(mixture_weights)
    dot_layer = keras.layers.Dot(axes=(1,-1))
    mixture_bpr_loss_val = dot_layer([mixture_weights, negative_bpr_K_val])

    return mixture_bpr_loss_val

def get_bpr_loss_func(K, num_samples=1000, sigma=1, noise='normal'):

    top_K_idx_func = partial(top_k_idx, k=K)
    perturbed_top_K = perturbed(top_K_idx_func,
                                    num_samples=num_samples,
                                    sigma=sigma,
                                    noise=noise,
                                    batched=True)

    negative_bpr_K = partial(negative_bpr_K_uncurried,K=K, perturbed_top_K_func=perturbed_top_K)

    return negative_bpr_K


def uncurried_penalized_bpr(y_true, y_pred, loss_func=poisson_nll, penalty=1.0, 
                            bpr_threshold=0.40, negative_bpr_K_func=None):
    
    negative_bpr_K_val = negative_bpr_K_func(y_true, y_pred)
    # threshold is violated when threshold is above bpr
    violate_threshold_flag = tf.cast(tf.greater(bpr_threshold,
                                                -negative_bpr_K_val),
                                     tf.float32)
    
    loss = loss_func(y_true, y_pred) + penalty * violate_threshold_flag *(bpr_threshold + negative_bpr_K_val)

    return loss

def get_penalized_bpr_loss_func(loss_func, K, penalty, bpr_threshold,
                                num_samples=1000, sigma=1, noise='normal'):
    negative_bpr_K_func = get_bpr_loss_func(K, num_samples, sigma, noise)
    penalized_bpr = partial(uncurried_penalized_bpr, loss_func=loss_func, penalty=penalty, 
                            bpr_threshold=bpr_threshold, 
                            negative_bpr_K_func=negative_bpr_K_func)

    return penalized_bpr

@tf.function
def mix_bpr(y_true, y_pred, negative_bpr_K_func=None):

    component_preds, mixture_weights = y_pred

    mixture_weights = tf.ones((component_preds.shape[0],1)) * tf.squeeze(mixture_weights)
    dot_layer = keras.layers.Dot(axes=(1,-1))
    mixture_pred = dot_layer([mixture_weights, component_preds])

    mixture_bpr_loss_val = negative_bpr_K_func(y_true, mixture_pred)

    return mixture_bpr_loss_val

@tf.function
def uncurried_penalized_mix_bpr(y_true, y_pred, loss_func=poisson_nll, penalty=1.0, 
                            bpr_threshold=0.40, negative_bpr_K_func=None):
    
    mixture_bpr_loss_val = mix_bpr(y_true, y_pred, negative_bpr_K_func=negative_bpr_K_func)

    # threshold is violated when threshold is above bpr
    violate_threshold_flag = tf.cast(tf.greater(bpr_threshold,
                                                -mixture_bpr_loss_val),
                                     tf.float32)
    
    
    loss = tf.math.reduce_sum(loss_func(y_true, y_pred),axis=-1) + penalty * violate_threshold_flag *(bpr_threshold + mixture_bpr_loss_val)

    return loss

def get_penalized_bpr_loss_func_mix(loss_func, K, penalty, bpr_threshold,
                                 num_samples=1000, sigma=1, noise='normal'):
    negative_bpr_K_func = get_bpr_loss_func(K, num_samples, sigma, noise)
    penalized_bpr = partial(uncurried_penalized_mix_bpr, loss_func=loss_func, penalty=penalty, 
                            bpr_threshold=bpr_threshold, 
                            negative_bpr_K_func=negative_bpr_K_func)

    return penalized_bpr