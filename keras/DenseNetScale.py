import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import initializers as initializations

#--------------------------------------------------------------------------------
#---- This class builds a custom layer for  DenseNet
#---- This layer is used for batch normalization, used in scaling the input
#---- Lean a set of weights, where output is the following
#---- out = in * gamma + beta
#---- This class is similar to https://github.com/flyyufelix/DenseNet-Keras/custom_layers.py

class Scale(Layer):

    # --------------------------------------------------------------------------------
    #---- Initialize the layer
    #---- axis - axis along which to normalize the tensor
    #---- momentum - for feature-wis normalization, momentum in computation of mean, st-d
    #---- weights - initialization weights. list of 2 numpy arrays
    #---- betaInit - name of the initialization function  for beta
    #---- gammaInit - name of the initialization function for gamma

    def __init__(self, weights = None, axis = -1, momentum = 0.9, betaInit='zero', gammaInit='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(betaInit)
        self.gamma_init = initializations.get(gammaInit)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    #--------------------------------------------------------------------------------
    #---- Builds the layer of the desired shape
    #---- inputShape - shape

    def build(self, inputShape):

        self.input_spec = [InputSpec(shape=inputShape)]
        shape = (int(inputShape[self.axis]),)

        self.gamma =  tf.Variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta =  tf.Variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
    #--------------------------------------------------------------------------------

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out =  tf.reshape(self.gamma, broadcast_shape) * x +  tf.reshape(self.beta, broadcast_shape)
        return out

    #--------------------------------------------------------------------------------

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#--------------------------------------------------------------------------------