from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfvan.pad import SamePad


@register_keras_serializable(package='TFVan')
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, patch_stride, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.pad = SamePad(self.patch_size)

        # noinspection PyAttributeOutsideInit
        self.proj = layers.Conv2D(
            self.embed_dim, kernel_size=self.patch_size, strides=self.patch_stride, name='proj')

        # noinspection PyAttributeOutsideInit
        self.norm = layers.BatchNormalization(name='norm', epsilon=1.001e-5)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.pad(inputs)
        outputs = self.proj(outputs)
        outputs = self.norm(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.pad.compute_output_shape(input_shape)
        output_shape = self.proj.compute_output_shape(output_shape)

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'patch_stride': self.patch_stride,
            'embed_dim': self.embed_dim
        })

        return config
