import keras
import tensorflow as tf
import tensorflow_probability as tfp
from metrics import cross_ratio_decision
from experiments import score_function_trick, overall_gradient_calculation

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

def location_specific_linear(input_shape, seed=360, activation='softplus'):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(name='linear_input', shape=input_shape),
            # convolution to turn H,S into 1,S
            # Filters = size oute convolutiput space
            # kernel_size = size of thon window
            # dataformat = channels_first means that the input shape is (batch_size, features, time)
            keras.layers.Conv1D(name='linear_convolution', filters=1, kernel_size=1, activation=activation, data_format='channels_first'),
            keras.layers.Flatten(name='linnear_flatten'),
        ]
    )
    return model


class DeprecatedMixtureWeightLayer(keras.layers.Layer):
    """Dumb layer that just returns mixture weights
    Constrained to unit norm
    """
    def __init__(self, num_components=2,):
        super().__init__()
        self.w = self.add_weight(name='mix_weights',
            shape=(num_components, 1),
            initializer="uniform",
            trainable=True,
        )
        
        self.softmax = keras.layers.Softmax(axis=0)

    def call(self, inputs):
        return self.softmax(self.w)
    

class LocationSpecificMixtureWeightLayer(keras.layers.Layer):
    """Dumb layer that just returns mixture weights
    Constrained to unit norm
    """
    def __init__(self, num_locations, num_components=2, **kwargs):
        super().__init__(**kwargs)
        self.w = self.add_weight(name='shared_mix_weights',
            shape=(num_components, num_locations),
            initializer="uniform",
            trainable=True,
        )
        
        self.softmax = keras.layers.Softmax(axis=0)

    def call(self, inputs):
        return self.softmax(self.w)
    
class LocationSpecificMixtureWeightLayerRightOrder(keras.layers.Layer):
    """Dumb layer that just returns mixture weights
    Constrained to unit norm
    """
    def __init__(self, num_locations, num_components=2, **kwargs):
        super().__init__(**kwargs)
        self.w = self.add_weight(name='shared_mix_weights',
            shape=(num_locations, num_components ),
            initializer="uniform",
            trainable=True,
        )
        
        self.softmax = keras.layers.Softmax(axis=1)

    def call(self, inputs):
        return self.softmax(self.w)
    
class CustomMixtureModel(keras.Model):
    """Does nothing. Just functional model."""

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
class CustomPenalizedMixtureDecisionModel(keras.Model):

    def __init__(self, name="mixture_decision_model",
                 num_features=50, num_locations=12,
                 member_model=location_specific_linear,
                 member_distribution=tfp.distributions.Poisson,
                 decision_func=cross_ratio_decision,
                 bpr_func=None,
                 bpr_threshold=0.55,
                 penalty=50,
                 objective_includes_likelihood=False,
                 objective_includes_bpr=False,
                 num_components=4,
                 num_score_func_samples=50,
                 seed=360,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.num_locations = num_locations
        self.member_model = member_model
        self.member_distribution = member_distribution
        self.decision_func = decision_func
        self.bpr_func = bpr_func
        self.bpr_threshold = bpr_threshold
        self.penalty = penalty
        self.objective_includes_bpr = objective_includes_bpr
        self.objective_includes_likelihood = objective_includes_likelihood
        self.num_components = num_components
        self.num_score_func_samples = num_score_func_samples
        self.seed = seed

        # TODO: seed is weird
        self.member_models = [member_model((num_features, num_locations),
                                            seed=self.seed+1000*m) for m in range(num_components)]

        # For flattening member predictions
        # target_shape(-1,1) means transform prediction into a whatever-by-1 tensor.
        # the final 1 becomes the component dimension
        self.reshape_member_preds = keras.layers.Reshape(name='mix_reshape', target_shape=(-1,1))
        # concat along the component dimension
        self.concat_member_preds =  keras.layers.Concatenate(name='mix_concat', axis=-1)

        self.mixture_weight_layer_K = LocationSpecificMixtureWeightLayer(num_locations, num_components=num_components)

        inputs_HS = keras.Input(shape=(num_features, num_locations), name='mix_input')
        self.mixture_weights_KS = self.mixture_weight_layer_K(inputs_HS)


        self.objective_tracker = keras.metrics.Mean(name="loss")
        

        return
    
    def _get_mixture_distribution(self, x_BHS):
     
        component_params_BSK, mixture_weights_SK = self._get_component_params_and_mix_weights(x_BHS)

        mixture_distribution = tfp.distributions.MixtureSameFamily(
                                    mixture_distribution=tfp.distributions.Categorical(probs=mixture_weights_SK),
                                    components_distribution = self.member_distribution(rate=component_params_BSK))
        return mixture_distribution


    
    def _get_component_params_and_mix_weights(self, x_BHS):
        """Helper function that does forward pass through to calculating model params"""

        # member predictions
        # TODO that seed aint right
        member_preds_kBS = [member(x_BHS) for member in self.member_models]
        
        reshaped_kBS1 = [self.reshape_member_preds(member) for member in member_preds_kBS]
        component_params_BSK = self.concat_member_preds(reshaped_kBS1)

        # my fault that model returns KS instead of SK
        mixture_weights_SK = tf.transpose(self.mixture_weights_KS, perm=[1,0])

        return component_params_BSK, mixture_weights_SK


    def call(self, x_BS):

        mixture_distribution = self._get_mixture_distribution(x_BS)
        
        sample_y_MBS = mixture_distribution.sample(self.num_score_func_samples)

        sample_decisions_MBS = self.decision_func(sample_y_MBS)
        expected_decisions_BS = tf.reduce_mean(sample_decisions_MBS, axis=0)

        return expected_decisions_BS
    
    def calc_loss_and_metrics(self, y_BS, mixture_distribution, expected_decisions_BS):
        observed_log_prob_BS = mixture_distribution.log_prob(y_BS)
        bpr_B = self.bpr_func(y_BS, expected_decisions_BS)

        loss_B = tf.zeros_like(bpr_B)

        if self.objective_includes_likelihood:
            loss_B -= tf.reduce_sum(observed_log_prob_BS, axis=-1)

        if self.objective_includes_bpr:
            violate_threshold_flag_B = tf.cast(tf.greater(self.bpr_threshold,
                                                        bpr_B),
                                                tf.float32)
            loss_B += self.penalty * violate_threshold_flag_B *(self.bpr_threshold - bpr_B)

        metric_dict = {'loss': tf.reduce_mean(loss_B, axis=0),
                       'bpr': tf.reduce_mean(bpr_B, axis=0),
                       'll': tf.reduce_mean(tf.reduce_sum(observed_log_prob_BS, axis=-1), axis=0)
                       }
        return loss_B, metric_dict
            


    def train_step(self, data):

        x_BHS, y_BS = data

        with tf.GradientTape() as jacobian_tape, tf.GradientTape() as loss_tape:
            mixture_distribution = self._get_mixture_distribution(x_BHS)
            import pdb;pdb.set_trace();
            sample_y_MBS = mixture_distribution.sample(self.num_score_func_samples)

            sample_log_likelihood_MBS = mixture_distribution.log_prob(sample_y_MBS)

            sample_decisions_MBS = self.decision_func(sample_y_MBS)
            expected_decisions_BS = tf.reduce_mean(sample_decisions_MBS, axis=0)

            loss_B, metrics = self.calc_loss_and_metrics(y_BS, mixture_distribution, expected_decisions_BS)


        # The lowercase "p" signifies that these are lists of length P, P = number of trainable variables
        jacobian_pMBS = jacobian_tape.jacobian(sample_log_likelihood_MBS, self.trainable_variables)
        param_gradient_pBS = score_function_trick(jacobian_pMBS, sample_decisions_MBS)
        loss_gradients_BS = loss_tape.gradient(loss_B, expected_decisions_BS)
        overall_gradient_p = [overall_gradient_calculation(g, loss_gradients_BS) for g in param_gradient_pBS]
        self.optimizer.apply(overall_gradient_p, self.trainable_variables)

        return metrics

def mixture_poissons(model, input_shape, num_components=2, seed=360):

    num_features, num_locations = input_shape

    member_models = []
    for c in range(num_components):
        member_models.append(model(input_shape, seed=seed+1000*c))

    inputs = keras.Input(shape=input_shape, name='mix_input')

    reshape_layer = keras.layers.Reshape(name='mix_reshape', target_shape=(-1,1))

    concat_layer = keras.layers.Concatenate(name='mix_concat',axis=-1)

    mixture_weight_layer = LocationSpecificMixtureWeightLayer(num_locations,name='mixture_weights', num_components=num_components)

    reshaped = [reshape_layer(member(inputs)) for member in member_models]
    concatted = concat_layer(reshaped)
    
    mixture_weights = mixture_weight_layer(inputs)

    # Call mixture layer to initialize
    #mixed = mixture_layer([mixture_weights, concatted])

    # outputs are NOT mixture, we need output of each component for loss
    model = CustomMixtureModel(name='mix_model',inputs=inputs,
                        outputs=[concatted, mixture_weights])
    
    return model, mixture_weights

def get_mixture(model, input_shape, num_components=2, seed=360):

    num_features, num_locations = input_shape

    member_models = []
    for c in range(num_components):
        member_models.append(model(input_shape, seed=seed+1000*c))

    # Define layers
    inputs = keras.Input(shape=input_shape, name='mix_input')
    reshape_layer = keras.layers.Reshape(name='mix_reshape', target_shape=(-1,1))
    concat_layer = keras.layers.Concatenate(name='mix_concat',axis=-1)
    add_layer = keras.layers.Add(name='add_const')


    mixture_weight_layer = LocationSpecificMixtureWeightLayerRightOrder(num_locations,name='mixture_weights', num_components=num_components)

    reshaped = [reshape_layer(member(inputs)) for member in member_models]
    concatted = concat_layer(reshaped)
    added = add_layer([concatted, tf.constant([1e-13])])
    
    mixture_weights = mixture_weight_layer(inputs)

    mixture_distribution_layer = tfp.layers.DistributionLambda(lambda params: 
        tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(probs=params[0]),
                                            components_distribution = tfp.distributions.Poisson(rate=params[1], validate_args=True)))
    
    outputs = mixture_distribution_layer([mixture_weights, added])

    # Call mixture layer to initialize
    #mixed = mixture_layer([mixture_weights, concatted])

    # outputs are NOT mixture, we need output of each component for loss
    model = CustomMixtureModel(name='mix_model',inputs=inputs,
                        outputs=[outputs])
    
    return model
