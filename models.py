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

def mixture_poissons(input_shape, num_components=2, seed=360):
    member_models = []
    for c in range(num_components):
        member_models.append(poisson_glm(input_shape, seed=seed+1000*c))

    inputs = keras.Input(shape=input_shape)

    reshape_layer = keras.layers.Reshape(target_shape=(-1,1))

    concat_layer = keras.layers.Concatenate(axis=-1)

    mixture_layer = keras.layers.Conv1D(filters=1, kernel_size=1, activation=None, use_bias=False,
                                        kernel_initializer='uniform', data_format='channels_last',
                                        kernel_constraint=keras.constraints.UnitNorm())
    

    reshaped = [reshape_layer(member(inputs)) for member in member_models]
    concatted = concat_layer(reshaped)
    
    # Call mixture layer to initialize
    mixed = mixture_layer(concatted)

    # outputs are NOT mixture, we need output of each component for loss
    model = keras.Model(inputs=inputs,
                        outputs=concatted)
    
    mixture_weights = mixture_layer.trainable_variables[0]
    
    return model, mixture_weights