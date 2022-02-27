import math
from keras import activations, layers
from keras.utils.conv_utils import normalize_tuple
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFVan')
class Attention(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')

        self.dilation_rate = Attention.dilation(self.kernel_size[0]), Attention.dilation(self.kernel_size[1])
        self.kernel_size_dw = self.dilation_rate[0] * 2 - 1, self.dilation_rate[1] * 2 - 1
        self.kernel_size_dwd = math.ceil(self.kernel_size[0] / self.dilation_rate[0]), \
                               math.ceil(self.kernel_size[1] / self.dilation_rate[1])

    @staticmethod
    def dilation(kernel):
        # Choose dilation rate with respect to minimum multiplications
        # import sympy
        # kernel, dilation = sympy.symbols('kernel,dilation', positive=True)
        # kernel1, kernel2 = 2 * dilation - 1, kernel / dilation
        # mults = kernel1 ** 2 + kernel2 ** 2
        # dmult = sympy.diff(mults, dilation)
        # sympy.solveset(dmult, dilation)

        root_k6_k4 = (kernel ** 6 / 1728 + kernel ** 4 / 65536) ** (1 / 2)
        root_k2_k6_k4 = 2 * (root_k6_k4 - kernel ** 2 / 256) ** (1 / 3)
        frac_k2_d6 = kernel ** 2 / (3 * root_k2_k6_k4)
        part_1 = (root_k2_k6_k4 - frac_k2_d6 + 1 / 16) ** (1 / 2)
        part_2 = (frac_k2_d6 - root_k2_k6_k4 + 1 / 8 + 1 / (32 * part_1)) ** (1 / 2)

        dilation = part_2 / 2 - part_1 / 2 + 1 / 8

        if not isinstance(dilation, float):
            raise ValueError(f'Can\'t estimate dilation for kernel size {kernel}')

        return round(dilation)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.dw = layers.DepthwiseConv2D(self.kernel_size_dw, padding='same', name='conv0')
        self.dwd = layers.DepthwiseConv2D(
            self.kernel_size_dwd, padding='same', dilation_rate=self.dilation_rate, name='conv_spatial')
        self.pw = layers.Conv2D(channels, 1, name='conv1')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        attn = self.dw(inputs)
        attn = self.dwd(attn)
        attn = self.pw(attn)
        outputs = inputs * attn

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})

        return config


@register_keras_serializable(package='TFVan')
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.proj1 = layers.Conv2D(channels, 1, name='proj_1')
        self.attend = Attention(21, name='spatial_gating_unit')
        self.proj2 = layers.Conv2D(channels, 1, name='proj_2')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.proj1(inputs)
        outputs = activations.gelu(outputs)
        outputs = self.attend(outputs)
        outputs = self.proj2(outputs)
        outputs = outputs + inputs

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
