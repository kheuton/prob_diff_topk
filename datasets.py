import numpy as np
import tensorflow as tf

def make_data(dist_S, H=50, T=500, seed=360):
    """Generate data from a list of distributions.
    
    Args:
        dist_S: list of distributions
        H: number of historical observations to be used as features
        T: number of timepoints
        seed: random seed
    Returns
        X_THS: numpy array of shape (T, H, S)
        y_TS: numpy array of shape (T, S)
    """

    S = len(dist_S)
    # In order to have T timepoints each with H historical observations, we need T+H samples 
    data_HT_S = np.zeros((H+T, S))
    for s, dist in enumerate(dist_S):
        random_state = np.random.RandomState(10000 * seed + s)
        data_HT_S[:, s] = dist.rvs(size=H+T, random_state=random_state)

    X_THS = np.array([data_HT_S[t:H+t,:] for t in range(T)], dtype=np.float32)
    y_TS = np.array([data_HT_S[H+t, :] for t in range(T)], dtype=np.float32)

    return X_THS, y_TS

def train_val_test_split(X, y, train_pct, test_pct):
    val_pct = 1-train_pct-test_pct

    assert(int(train_pct*len(X)) == train_pct*len(X))
    assert(int(val_pct*len(X)) == val_pct*len(X))
    assert(int(test_pct*len(X)) == test_pct*len(X))

    train_X = X[:int(train_pct*len(X))]
    train_y = y[:int(train_pct*len(X))]
    val_X = X[int(train_pct*len(X)):int((train_pct+val_pct)*len(X))]
    val_y = y[int(train_pct*len(X)):int((train_pct+val_pct)*len(X))]
    test_X = X[int((train_pct+val_pct)*len(X)):]
    test_y = y[int((train_pct+val_pct)*len(X)):]

    assert(len(train_X) + len(val_X) + len(test_X) == len(X))

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

def tensorflow_dataset(X, y, seed=360, batch_size=32):

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size)

    return dataset