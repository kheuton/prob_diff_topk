import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from metrics import mix_bpr, mixture_poi_loss
from datasets import to_numpy

def training_loop(model, loss_func, optimizer, num_epochs, train_dataset, val_dataset, negative_bpr_K, verbose=False):

    losses = {}
    losses['train'] = {}
    losses['val'] ={}
    losses['train']['loss']=[]
    losses['train']['nll']=[]
    losses['train']['bpr']=[]
    losses['val']['loss']=[]
    losses['val']['nll']=[]
    losses['val']['bpr']=[]


    
    train_X_THS, train_y_TS = to_numpy(train_dataset)
    val_X_THS, val_y_TS = to_numpy(val_dataset)

    for epoch in range(num_epochs):

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                y_preds = model(x_batch_train, training=True)
                loss_value = loss_func(y_batch_train, y_preds)
                
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        y_preds = model(train_X_THS)
        loss = loss_func(train_y_TS, y_preds)
        loss = tf.reduce_mean(loss)
        bpr = mix_bpr(train_y_TS, y_preds, negative_bpr_K_func=negative_bpr_K)
        nll = mixture_poi_loss(train_y_TS, y_preds)
        nll = tf.reduce_mean(tf.reduce_sum(nll,axis=-1))
        losses['train']['loss'].append(loss)
        losses['train']['nll'].append(nll)
        losses['train']['bpr'].append(bpr)
        if verbose:
            print(f'{epoch}: {loss}')
        y_preds = model(val_X_THS)
        loss = loss_func(val_y_TS, y_preds)
        loss = tf.reduce_mean(loss)
        bpr = mix_bpr(val_y_TS, y_preds, negative_bpr_K_func=negative_bpr_K)
        nll = mixture_poi_loss(val_y_TS, y_preds)
        nll = tf.reduce_mean(tf.reduce_sum(nll,axis=-1))
        losses['val']['loss'].append(loss)
        losses['val']['nll'].append(nll)
        losses['val']['bpr'].append(bpr)

    return losses

def overall_gradient_calculation(gradient_BSp, decision_gradient_BS):
    num_param_dims = tf.rank(gradient_BSp)-2

    decision_gradient_BSp = tf.reshape(decision_gradient_BS, decision_gradient_BS.shape + [1]*num_param_dims.numpy())

    overall_gradient_BSp = gradient_BSp*decision_gradient_BSp

    # sum over batch and location
    overall_gradient = tf.reduce_sum(overall_gradient_BSp, axis=[0,1])
    return overall_gradient
    
#@tf.function
def score_function_trick(jacobian_MBSp, decision_MBS):
    num_param_dims = tf.rank(jacobian_MBSp)-3
    # expand decision to match jacobian
    decision_MBSp = tf.reshape(decision_MBS, decision_MBS.shape + [1]*num_param_dims.numpy())

    scaled_jacobian_MBSp = jacobian_MBSp*decision_MBSp

    # average over sample dims
    param_gradient_BSp = tf.reduce_mean(scaled_jacobian_MBSp, axis=0)

    return param_gradient_BSp

def training_loop_score_function_trick(model, optimizer, num_epochs, train_dataset, val_dataset,
                                       decision_func, bpr_func,
                                       objective_includes_likelihood=False,
                                       objective_includes_bpr=False,
                                       bpr_threshold=0.55,
                                       penalty=5000,
                                       num_score_func_samples=10,
                                       verbose=False):

    losses = {}
    losses['train'] = {}
    losses['val'] ={}
    losses['train']['loss']=[]
    losses['train']['nll']=[]
    losses['train']['bpr']=[]
    losses['val']['loss']=[]
    losses['val']['nll']=[]
    losses['val']['bpr']=[]


    
    train_X_THS, train_y_TS = to_numpy(train_dataset)
    val_X_THS, val_y_TS = to_numpy(val_dataset)

    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch}')

        for step, (x_BHS, y_BS) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as jacobian_tape, tf.GradientTape() as loss_tape:
                prob_params_BSK, mixture_weights_KS = model(x_BHS, training=True)

                # my fault that model returns KS instead of SK
                mixture_weights_SK = tf.transpose(mixture_weights_KS, perm=[1,0])

                #  could create a custom model class that returns 
                # the appropriate tfp.dist given outputs
                mix = tfp.distributions.MixtureSameFamily(
                    mixture_distribution=tfp.distributions.Categorical(probs=mixture_weights_SK),
                    components_distribution = tfp.distributions.Poisson(rate=prob_params_BSK))

                # add constant to avoid log 0
                sample_y_MBS = mix.sample(num_score_func_samples)+1e-13

                sample_log_probs_MBS = mix.log_prob(sample_y_MBS)

                sample_decisions_MBS = decision_func(sample_y_MBS)
                expected_decisions_BS = tf.reduce_mean(sample_decisions_MBS, axis=0)

                bpr_B = bpr_func(y_BS, expected_decisions_BS)
                observed_log_prob_BS = mix.log_prob(y_BS)

                loss_B = tf.zeros_like(bpr_B)

                if objective_includes_likelihood:
                    loss_B -= tf.reduce_sum(observed_log_prob_BS, axis=-1)
                if objective_includes_bpr:
                    violate_threshold_flag_B = tf.cast(tf.greater(bpr_threshold,
                                                                bpr_B),
                                                        tf.float32)
                    loss_B += penalty * violate_threshold_flag_B *(bpr_threshold - bpr_B)


            # The lowercase "p" signifies that these are lists of length P, P = number of trainable variables
            jacobian_pMBS = jacobian_tape.jacobian(sample_log_probs_MBS, model.trainable_weights)
            param_gradient_pBS = [score_function_trick(j, sample_decisions_MBS) for j in jacobian_pMBS]

            loss_gradients_BS = loss_tape.gradient(loss_B, expected_decisions_BS)
            overall_gradient = [overall_gradient_calculation(g, loss_gradients_BS) for g in param_gradient_pBS]

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(overall_gradient, model.trainable_weights))

            losses['train']['loss'].append(tf.reduce_mean(loss_B))
            losses['train']['nll'].append(tf.reduce_mean(tf.reduce_sum(observed_log_prob_BS, axis=-1)))
            losses['train']['bpr'].append(tf.reduce_mean(bpr_B))
            
            if verbose:
                # print all metrics
                print(f'Loss: {losses["train"]["loss"][-1]}')
                print(f'NLL: {losses["train"]["nll"][-1]}')
                print(f'BPR: {losses["train"]["bpr"][-1]}')

        '''
        y_preds = model(train_X_THS)
        loss = loss_func(train_y_TS, y_preds)
        loss = tf.reduce_mean(loss)
        bpr = mix_bpr(train_y_TS, y_preds, negative_bpr_K_func=negative_bpr_K)
        nll = mixture_poi_loss(train_y_TS, y_preds)
        nll = tf.reduce_mean(tf.reduce_sum(nll,axis=-1))
        losses['train']['loss'].append(loss)
        losses['train']['nll'].append(nll)
        losses['train']['bpr'].append(bpr)
        if verbose:
            print(f'{epoch}: {loss}')
        y_preds = model(val_X_THS)
        loss = loss_func(val_y_TS, y_preds)
        loss = tf.reduce_mean(loss)
        bpr = mix_bpr(val_y_TS, y_preds, negative_bpr_K_func=negative_bpr_K)
        nll = mixture_poi_loss(val_y_TS, y_preds)
        nll = tf.reduce_mean(tf.reduce_sum(nll,axis=-1))
        losses['val']['loss'].append(loss)
        losses['val']['nll'].append(nll)
        losses['val']['bpr'].append(bpr)'''

    return losses
    