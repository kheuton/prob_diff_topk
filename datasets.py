import numpy as np
import tensorflow as tf
from distributions import ZeroInflatedDist, QuantizedNormal

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
        random_state = np.random.RandomState(10000 * seed + s*123456)
        data_HT_S[:, s] = dist.rvs(size=H+T, random_state=random_state)

    X_THS = np.array([data_HT_S[t:H+t,:] for t in range(T)], dtype=np.float32)
    y_TS = np.array([data_HT_S[H+t, :] for t in range(T)], dtype=np.float32)

    return X_THS, y_TS

def example_datasets(H, T, dist_S=None, seed=360, batch_size=None, train_pct=0.6, test_pct=0.2,
                      return_dists=False, return_numpy=False):

    if batch_size is None:
        batch_size=T

    if dist_S is None:
        consistent_4 = [QuantizedNormal(7, 0.1) for _ in range(4)]
        highvar_4 = [ZeroInflatedDist(QuantizedNormal(10, 0.1), 1-0.7) for _ in range(4)]
        powerball_4 = [ZeroInflatedDist(QuantizedNormal(100, 0.1), 0.9) for _ in range(4)]
        dist_S = consistent_4 + highvar_4 +powerball_4


    X_THS, y_TS = make_data(dist_S, H=H, T=T, seed=seed)

    # check that each final history is equal to the previous observation
    for t in range(H, T):
        assert(np.all(X_THS[t, H-1, :] == y_TS[t-1, :]))

    # check random point in history
    for t in range(H, T):
        h = np.random.randint(0, H)
        assert(np.all(X_THS[t, h, :] == y_TS[t-(H-h), :]))

    (train_X_THS, train_y_TS), \
    (val_X_THS, val_y_TS), \
    (test_X_THS, test_y_TS) = train_val_test_split(X_THS, y_TS, train_pct=train_pct, test_pct=test_pct)

    train_dataset = tensorflow_dataset(train_X_THS, train_y_TS, seed=seed+200,batch_size=batch_size)
    val_dataset = tensorflow_dataset(val_X_THS, val_y_TS, seed=seed+300,batch_size=batch_size)
    test_dataset = tensorflow_dataset(test_X_THS, test_y_TS, seed=seed+300,batch_size=batch_size)

    return_objs = (train_dataset, val_dataset, test_dataset)

    if return_dists:
        return_objs += (dist_S,)
    if return_numpy:
        return_objs += ((train_X_THS, train_y_TS), (val_X_THS, val_y_TS), (test_X_THS, test_y_TS))

    return return_objs

def train_val_test_split(X, y, train_pct, test_pct):
    val_pct = 1-train_pct-test_pct

    assert(np.isclose(int(train_pct*len(X)), train_pct*len(X), atol=1e-3))
    assert(np.isclose(int(val_pct*len(X)), val_pct*len(X), atol=1e-3))
    assert(np.isclose(int(test_pct*len(X)), test_pct*len(X), atol=1e-3))

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

def to_numpy(dataset):
    x_array = []
    y_array = []

    for x, y in dataset:
        x_array.append(x.numpy())
        y_array.append(y.numpy())

    x_array = np.squeeze(np.array(x_array))
    y_array = np.squeeze(np.array(y_array))

    return x_array, y_array