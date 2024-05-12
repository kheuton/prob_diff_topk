import os
from functools import partial

import keras

from datasets import example_datasets, to_numpy
from models import mixture_poissons,location_specific_linear
from metrics import mixture_poi_loss, get_bpr_loss_func, mix_bpr, get_penalized_bpr_loss_func_mix
from experiments import training_loop
from plotting_funcs import plot_losses, plot_frontier, plot_component_histograms

def main(seed=None, num_components=None, learning_rate=None, epochs=None, outdir=None,
         penalty=None, threshold=None, K=None, do_only=None):
    # tracts/distributions
    S=12
    # history/features
    H = 3
    # total timepoints
    T= 500


    train_dataset, val_dataset, test_dataset = example_datasets(H, T, seed=seed)
    train_X_THS, train_y_TS = to_numpy(train_dataset)
    val_X_THS, val_y_TS = to_numpy(val_dataset)
    input_shape = (H,S)

    negative_bpr_K = get_bpr_loss_func(K)

    if do_only:
      # NLL loss only
      mix_model_nll, _  = mixture_poissons(location_specific_linear, input_shape, num_components=num_components)
      optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
      losses_nll = training_loop(mix_model_nll, mixture_poi_loss, optimizer, epochs,
                                  train_dataset, val_dataset, negative_bpr_K,
                                verbose=True)
      mix_model_nll.save(os.path.join(outdir, 'nll_only_model'))
      plot_losses(losses_nll, title_add='NLL Only', save_dir=outdir, file_add='nll')

      y_preds_val = mix_model_nll(val_X_THS)
      plot_component_histograms(y_preds_val, title_add='NLL Only', save_dir=outdir, file_add='nll')

      # BPR loss only
      mix_model_bpr, _  = mixture_poissons(location_specific_linear, input_shape, num_components=num_components)
      optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
      bpr_only_loss = partial(mix_bpr, negative_bpr_K_func=negative_bpr_K)
      losses_bpr = training_loop(mix_model_bpr, bpr_only_loss, optimizer, epochs, train_dataset, val_dataset, negative_bpr_K,
                                verbose=True)
      mix_model_bpr.save(os.path.join(outdir, 'bpr_only_model'))
      plot_losses(losses_bpr, title_add='BPR Only', save_dir=outdir, file_add='bpr')

      y_preds_val = mix_model_bpr(val_X_THS)
      plot_component_histograms(y_preds_val, title_add='BPR Only', save_dir=outdir, file_add='bpr')

    # Penalized loss
    mix_model_penalized, _  = mixture_poissons(location_specific_linear, input_shape, num_components=num_components)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    penalized_bpr_loss = get_penalized_bpr_loss_func_mix(mixture_poi_loss, K, penalty, threshold)
    losses_penalized = training_loop(mix_model_penalized, penalized_bpr_loss, optimizer,
                                      epochs, train_dataset, val_dataset, negative_bpr_K,
                                        verbose=True)
    mix_model_penalized.save(os.path.join(outdir, 'penalized_model'))
    plot_losses(losses_penalized, title_add='Penalized', save_dir=outdir, file_add='penalized')
    y_preds_val = mix_model_penalized(val_X_THS)
    plot_component_histograms(y_preds_val, title_add='Penalized', save_dir=outdir, file_add='penalized')


    if do_only:
      plot_frontier(losses_nll, losses_bpr, losses_penalized, savedir=outdir)
    

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--do_only', action='store_true')
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_components', type=int, default=4)
    parser.add_argument('--seed', type=int, default=360)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--penalty', type=float, required=True)
    parser.add_argument('--threshold', type=float, required=True)

    args = parser.parse_args()

    main(**vars(args))