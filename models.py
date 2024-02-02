import keras


def poisson_glm(input_shape, seed=360):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            # convolution to turn H,S into 1,S
            # Filters = size oute convolutiput space
            # kernel_size = size of thon window
            # dataformat = channels_first means that the input shape is (batch_size, features, time)
            keras.layers.Conv1D(filters=1, kernel_size=1, activation='softplus', data_format='channels_first'),
            keras.layers.Flatten(),
        ]
    )
    return model