import math
from keras import activations, layers, initializers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFVan')
class MLP(layers.Layer):
    def __init__(self, ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.ratio = ratio
        self.dropout = dropout

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        inner_channels = int(channels * self.ratio)

        # noinspection PyAttributeOutsideInit
        self.pw1 = layers.Conv2D(
            inner_channels, 1, name='fc1',
            kernel_initializer=initializers.RandomNormal(0., math.sqrt(2. / inner_channels)))

        # noinspection PyAttributeOutsideInit
        self.dw1 = layers.DepthwiseConv2D(
            3, padding='same', name='dwconv.dwconv',
            kernel_initializer=initializers.RandomNormal(0., math.sqrt(2. / 9)))

        # noinspection PyAttributeOutsideInit
        self.pw2 = layers.Conv2D(
            channels, 1, name='fc2',
            kernel_initializer=initializers.RandomNormal(0., math.sqrt(2. / channels)))

        # noinspection PyAttributeOutsideInit
        self.drop = layers.Dropout(self.dropout)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.pw1(inputs)
        outputs = self.dw1(outputs)
        outputs = activations.gelu(outputs)
        outputs = self.drop(outputs)
        outputs = self.pw2(outputs)
        outputs = self.drop(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            'dropout': self.dropout
        })

        return config
