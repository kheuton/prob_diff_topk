import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

def plot_hexagon_grid(data, vmin=None, vmax=None, title=None):
    """
    Generate a grid of hexagons with colors corresponding to the values of an array.

    Parameters:
    - data: 1D array of numbers.

    Returns:
    - None (displays the plot).
    """
    # Calculate the size of the hexagon grid
    size = int(np.sqrt(len(data)))+1

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
    


def sample_and_plot(dist_N, size=1000, seed=360, average=False,median=False, index=-1, title=None):
    random_state = np.random.RandomState(seed)
    # Sample from each distribution
    data = np.array([d.rvs(size=size, random_state=random_state) for d in dist_N])
    vmin=np.min(data)
    vmax=np.max(data)

    if average or median:
        if average:
            data = np.mean(data, axis=1)
        else:
            data = np.median(data, axis=1)
        plot_hexagon_grid(data, title=title)
    else:
        data = np.array([samples[index] for samples in data])
        plot_hexagon_grid(data, vmin=vmin, vmax=vmax, title=title)


