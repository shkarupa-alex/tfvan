from keras import layers, initializers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfvan.attn import SpatialAttention
from tfvan.drop import DropPath
from tfvan.mlp import MLP


@register_keras_serializable(package='TFVan')
class Block(layers.Layer):
    def __init__(self, mlp_ratio, mlp_drop, path_drop, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.mlp_ratio = mlp_ratio
        self.mlp_drop = mlp_drop
        self.path_drop = path_drop

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.norm1 = layers.BatchNormalization(name='norm1', epsilon=1.001e-5)

        # noinspection PyAttributeOutsideInit
        self.attend = SpatialAttention(name='attn')

        # noinspection PyAttributeOutsideInit
        self.drop_path = DropPath(self.path_drop)

        # noinspection PyAttributeOutsideInit
        self.norm2 = layers.BatchNormalization(name='norm2', epsilon=1.001e-5)

        # noinspection PyAttributeOutsideInit
        self.mlp = MLP(self.mlp_ratio, self.mlp_drop)

        # noinspection PyAttributeOutsideInit
        self.scale1 = self.add_weight(
            'layer_scale_1', shape=[1, 1, 1, channels], initializer=initializers.Constant(1e-2),
            trainable=True, dtype=self.dtype)

        # noinspection PyAttributeOutsideInit
        self.scale2 = self.add_weight(
            'layer_scale_2', shape=[1, 1, 1, channels], initializer=initializers.Constant(1e-2),
            trainable=True, dtype=self.dtype)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        residual1 = self.norm1(inputs)
        residual1 = self.attend(residual1)
        residual1 *= self.scale1
        residual1 = self.drop_path(residual1)

        outputs = inputs + residual1

        residual2 = self.norm2(outputs)
        residual2 = self.mlp(residual2)
        residual2 *= self.scale2
        residual2 = self.drop_path(residual2)

        outputs = outputs + residual2

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_ratio': self.mlp_ratio,
            'mlp_drop': self.mlp_drop,
            'path_drop': self.path_drop
        })

        return config
