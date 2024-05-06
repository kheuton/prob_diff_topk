import numpy as np
import pandas as pd
import scipy.stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import tensorflow as tf

def plot_component_histograms(y_preds, true_values=[0,7,10,100],title_add='', save_dir=None, file_add=''):
    """Graph histogram of predictions

    Args:
        y_preds: Tuple of component predictions and mixture weights
            component predictions is batch-by-locations-by-components
            mixture weights is 
    """
    component_preds, mixture_weights = y_preds

    print(mixture_weights.shape)

    # Get the number of locations and mixture components
    num_locations = component_preds.shape[1]
    num_components = component_preds.shape[2]

    assert(len(true_values)==num_components)

    location_type_indices = [(0,4), (4, 8), (8, 12)]
    location_type_names = ['Consistent_7', '10_or_0', 'Lottery_100']

    # Iterate over the locations
    for indices, name in zip(location_type_indices, location_type_names):
        loc_min, loc_max = indices

        weights = mixture_weights[:, loc_min:loc_max]

        # Create a new figure for each location
        fig, ax = plt.subplots()
        n_vals = []

        # Iterate over the mixture components
        for component_idx in range(num_components):
            mix_weight = np.mean(weights[component_idx,:])
            # Get the predictions for the current location and component
            predictions = component_preds[:, loc_min:loc_max, component_idx]
            # make predictions 1d
            predictions = predictions.numpy().flatten()
            true_val = true_values[component_idx]

            # plot histogram with bins from 0 to 12 every 0.1
            bins = np.arange(-1, 1+int(np.max( component_preds[:, loc_min:loc_max,:])), 0.5)
            n_pred, _, _ = ax.hist(predictions, bins=bins, 
                                   label=f'Component {component_idx+1}: {mix_weight*100:.1f}%',
                                   density=True,
                                   alpha=0.8)
            # Plot histogram from a  poisson model with mean 7
            #n_true, _, _ = ax.hist(np.random.poisson(true_val, 1000),
            #          bins=range(120), label=f'Poisson({true_val})', density=True, alpha=0.5)
            n_vals.append(n_pred)
            #n_vals.append(n_true)

        #plt.ylim(0, max([max(n) for n in n_vals]))
        # Add labels and legend
        plt.xlabel('Value')
        plt.ylabel('Prediction')
        plt.title(f'{name} {title_add}')
        plt.legend()


        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'{name}_hist_{file_add}.png'))
        # Show the plot
        plt.show()

def plot_winners(y_preds, K, title_add='',save_dir=None, file_add=''):

    plt.figure()
    component_preds, mixture_weights = y_preds

    mixture_pred = tf.einsum('ijk,kj->ij', component_preds, mixture_weights)

    T, S = mixture_pred.shape

    # mixture pred is batch by locations
    # plot bar graph of how often each location is in top K
    winners = np.argsort(mixture_pred, axis=1)[:, -K:]
    counts = np.zeros((S,))
    for i in range(S):
        counts[i] = np.sum(winners == i)

    # plot frequencies instead of counts
    counts = counts / T

    colors = ['blue']*4 + ['green']*4 + ['red']*4

    plt.bar(np.arange(S), counts, color=colors)
    
    
    plt.xlabel("Location")
    plt.ylabel("Frequency in top K")
    plt.title(f"Frequency in top {K} by location {title_add}")


    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'winners_{file_add}.png'))
    plt.show()


    return

def plot_losses(losses, title_add='', save_dir=None, file_add=''):
    # plot loss and metrics from history
    plt.figure()
    plt.plot(losses['train']['loss'])
    plt.plot(losses['val']['loss'])
    plt.title(f'Model loss {title_add}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'loss_{file_add}.png'))
    plt.show()

    
    plt.figure()
    plt.plot(losses['train']['nll'])
    plt.plot(losses['val']['nll'])
    plt.title(f'Model NLL {title_add}')
    plt.ylabel('NLL')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'nll_{file_add}.png'))
    plt.show()

    
    plt.figure()
    plt.plot(losses['train']['bpr'])
    plt.plot(losses['val']['bpr'])
    plt.title(f'Model Negative BPR {title_add}')
    plt.ylabel('Negative BPR')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'bpr_{file_add}.png'))
    plt.show()


def plot_frontier(losses_nll, losses_bpr, losses_penalized, savedir=None):
    plt.figure()
    plt.plot(-np.array(losses_nll['val']['nll'][-1]), -np.array(losses_nll['val']['bpr'][-1]), marker='o', label='NLL')
    plt.plot(-np.array(losses_bpr['val']['nll'][-1]), -np.array(losses_bpr['val']['bpr'][-1]), marker='o', label='BPR')
    plt.plot(-np.array(losses_penalized['val']['nll'][-1]), -np.array(losses_penalized['val']['bpr'][-1]), marker='o', label='Penalized')
    plt.title('Validation Frontier')
    plt.legend()
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f'val_frontier.png'))
    plt.show()

    plt.figure()
    plt.plot(-np.array(losses_nll['train']['nll'][-1]), -np.array(losses_nll['train']['bpr'][-1]), marker='o', label='NLL')
    plt.plot(-np.array(losses_bpr['train']['nll'][-1]), -np.array(losses_bpr['train']['bpr'][-1]), marker='o', label='BPR')
    plt.plot(-np.array(losses_penalized['train']['nll'][-1]), -np.array(losses_penalized['train']['bpr'][-1]), marker='o', label='Penalized')
    plt.title('Training Frontier')
    plt.legend()
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f'train_frontier.png'))
    plt.show()

    return


def plot_hexagon_grid(data, vmin=None, vmax=None, title=None):
    """
    Generate a grid of hexagons with colors corresponding to the values of an array.

    Parameters:
    - data: 1D array of numbers.

    Returns:
    - None (displays the plot).
    """
    # Calculate the size of the hexagon grid
    size = int(np.ceil(np.sqrt(len(data))))

    # Create a hexagon grid
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Calculate hexagon parameters
    radius = 1
    dx = 3/2 * radius
    dy = np.sqrt(3) * radius

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,)
    sm.set_array(data)
    if vmin is not None:
        sm.set_clim(vmin=vmin)
    if vmax is not None:
        sm.set_clim(vmax=vmax)

    cbar = plt.colorbar(sm, orientation='vertical', fraction=0.046, pad=0.04, ax=ax)
    cbar.set_label('Values')

    # Plot hexagons with colors corresponding to the values
    data_idx = -1
    for i in range(size):
        for j in range(size):
            data_idx += 1
            if data_idx >= len(data):
                break
            x = j * dx
            y = -i * dy if j % 2 == 0 else -i * dy - dy/2

            color = data[data_idx]
            orientation = 45/2  # Rotate every other column
            # get color from sm
            color = sm.to_rgba(color)
            hexagon = RegularPolygon((x, y), numVertices=6, radius=radius, orientation=orientation, edgecolor='k', facecolor=color)
            ax.add_patch(hexagon)

    

    # Set x and y limits
    ax.set_xlim(-dx, (size - 1) * dx + dx)
    ax.set_ylim(-(size - 1) * dy - dy, dy)
    if title is not None:   
        plt.title(title)
    plt.show()
    #title
    


def sample_and_plot(dist_N, size=1000, seed=360, average=False,median=False, index=-1, title=None, vmin=None, vmax=None):
    random_state = np.random.RandomState(seed)
    # Sample from each distribution
    data = np.array([d.rvs(size=size, random_state=random_state) for d in dist_N])



    if average or median:
        if average:
            data = np.mean(data, axis=1)
        else:
            data = np.median(data, axis=1)
    else:
        if vmin is None:
            vmin=np.min(data)
        if vmax is None:
            vmax=np.max(data)
        data = np.array([samples[index] for samples in data])
    plot_hexagon_grid(data, vmin=vmin, vmax=vmax, title=title)


def plot_dist(dist, title=''):
    # Generate random samples from the custom distribution
    samples = dist.rvs(size=1000)

    # Calculate mean and median
    distribution_mean = np.mean(samples)
    distribution_median = np.median(samples)

    # Plot the histogram of samples
    plt.hist(samples, bins=50, density=True, alpha=0.5, color='blue', label='Custom Distribution')

    # Display mean and median on the plot
    plt.axvline(distribution_mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {distribution_mean:.2f}')
    plt.axvline(distribution_median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {distribution_median:.2f}')


    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


