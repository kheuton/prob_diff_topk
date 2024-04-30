import numpy as np
import tensorflow as tf

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
    