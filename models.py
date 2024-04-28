import keras
import tensorflow as tf

class LocationSpecificLinearSoftplus(keras.layers.Layer):
    def __init__(self):
        super(LocationSpecificLinearSoftplus, self).__init__()

    def build(self, input_shape):
        # weights same size as inputs
        self.kernel = self.add_weight("kernel",
                                      shape=input_shape[1:],
                                      initializer="glorot_normal",
                                      trainable=True)

    def call(self, inputs):
        # Element-wise multiplication
        multiplied = tf.multiply(inputs, self.kernel)
        # Sum over second-to-last dimension
        summed = tf.reduce_sum(multiplied, axis=-2)
        output = tf.nn.softplus(summed)
        return output


def poisson_glm_tract_specific_weights(input_shape, seed=360):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            LocationSpecificLinearSoftplus(),
            keras.layers.Flatten(),
        ]
    )
    return model

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


class MixtureWeightLayer(keras.layers.Layer):
    """Dumb layer that just returns mixture weights
    Constrained to unit norm
    """
    def __init__(self, num_components=2,):
        super().__init__()
        self.w = self.add_weight(
            shape=(num_components, 1),
            initializer="uniform",
            trainable=True,
        )
        
        self.softmax = keras.layers.Softmax(axis=0)

    def call(self, inputs):
        return self.softmax(self.w)
    
class CustomMixtureModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            print(len(y_pred))
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def mixture_poissons(model, input_shape, num_components=2, seed=360):
    member_models = []
    for c in range(num_components):
        member_models.append(model(input_shape, seed=seed+1000*c))

    inputs = keras.Input(shape=input_shape)

    reshape_layer = keras.layers.Reshape(target_shape=(-1,1))

    concat_layer = keras.layers.Concatenate(axis=-1)

    mixture_weight_layer = MixtureWeightLayer(num_components=num_components)

    reshaped = [reshape_layer(member(inputs)) for member in member_models]
    concatted = concat_layer(reshaped)
    
    mixture_weights = mixture_weight_layer(inputs)

    # Call mixture layer to initialize
    #mixed = mixture_layer([mixture_weights, concatted])

    # outputs are NOT mixture, we need output of each component for loss
    model = CustomMixtureModel(inputs=inputs,
                        outputs=[concatted, mixture_weights])
    
    return model, mixture_weights

def mixture_poissons2(input_shape, num_components=2, seed=360):
    member_models = []
    for c in range(num_components):
        member_models.append(poisson_glm(input_shape, seed=seed+1000*c))

    inputs = keras.Input(shape=input_shape)

    reshape_layer = keras.layers.Reshape(target_shape=(-1,1))

    concat_layer = keras.layers.Concatenate(axis=-1)

    mixture_weight_layer = MixtureWeightLayer(num_components=num_components)

    reshaped = [reshape_layer(member(inputs)) for member in member_models]
    concatted = concat_layer(reshaped)
    
    mixture_weights = mixture_weight_layer(inputs)
    add_weight_layer = keras.layers.Concatenate(axis=0)
    concatted = add_weight_layer([concatted, tf.expand_dims(tf.transpose(mixture_weights,(1,0)),axis=0)])

    # Call mixture layer to initialize
    #mixed = mixture_layer([mixture_weights, concatted])

    # outputs are NOT mixture, we need output of each component for loss
    model = CustomMixtureModel(inputs=inputs,
                        outputs=[concatted])
    
    return model, mixture_weights