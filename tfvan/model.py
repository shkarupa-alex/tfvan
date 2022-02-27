import numpy as np
import tensorflow as tf
from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.utils import data_utils, layer_utils
from tfvan.block import Block
from tfvan.embed import PatchEmbedding
from tfvan.norm import LayerNorm

BASE_URL = 'https://github.com/shkarupa-alex/tfvan/releases/download/1.0.0/{}.h5'
WEIGHT_HASHES = {
    'van_tiny': 'af4693df46f37535b5f7c104b342fe97cf14c706e4615af1c8b1f41824f92a6a',
    'van_small': 'ccc1d0a08b3b01d1cd8a474415c2c5aaf918a5c55c5c2ca3ee7a307c7973b9a6',
    'van_base': 'b6575016e1010047413d28977daeee94589f70909ee1de8cd0bee11337a7eb37',
    'van_large': '0cf52f5cb2c7aea84667ee3f7bb897ad8c1fbe3057fe99211a583def349696c1',
}


def Van(embed_dims, mlp_ratios, depths, drop_rate=0., path_drop=0.1, model_name='van',
        include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000,
        classifier_activation='softmax'):
    """Instantiates the Visual Attention Network architecture.

    Args:
      embed_dims: patch embedding dimensions.
      mlp_ratios: ratio of mlp hidden units to embedding units.
      depths: depth of each VAN stage.
      drop_rate: dropout rate.
      path_drop: stochastic depth rate
      model_name: model name.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.
      input_tensor: tensor (i.e. output of `layers.Input()`) to use as image input for the model.
      input_shape: shape tuple without batch dimension. Used to create input layer if `input_tensor` not provided.
      pooling: optional pooling mode for feature extraction when `include_top` is `False`.
        - `None` means that the output of the model will be the 3D tensor output of the last layer.
        - `avg` means that global average pooling will be applied to the output of the last layer, and thus the output
          of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True.
      classifier_activation: the activation function to use on the "top" layer. Ignored unless `include_top=True`.
        When loading pretrained weights, `classifier_activation` can only be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and 1000 != classes:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000.')

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor is not None:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format='channel_last',
        require_flatten=False,
        weights=weights)

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype='float32')
    else:
        image = layers.Input(shape=input_shape)

    path_drops = np.linspace(0., path_drop, sum(depths))

    # Define model pipeline
    x = image
    for i in range(len(depths)):
        patch_size = 7 if 0 == i else 3
        patch_stride = 4 if 0 == i else 2
        patch_name = f'patch_embed{i + 1}'
        x = PatchEmbedding(
            patch_size=patch_size, patch_stride=patch_stride, embed_dim=embed_dims[i], name=patch_name)(x)

        for j in range(depths[i]):
            path_drop = path_drops[sum(depths[:i]) + j]
            block_name = f'block{i + 1}.{j}'
            x = Block(
                mlp_ratio=mlp_ratios[i], mlp_drop=drop_rate, path_drop=path_drop, name=block_name)(x)

        x = LayerNorm(name=f'norm{i + 1}')(x)

    if include_top or pooling in {None, 'avg'}:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    else:
        raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {pooling}')

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, name='head')(x)
    x = layers.Activation(classifier_activation, dtype='float32', name='pred')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if 'imagenet' == weights and model_name in WEIGHT_HASHES:
        weights_url = BASE_URL.format(model_name)
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfvan')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    last_layer = 'norm4'
    if pooling == 'avg':
        last_layer = 'avg_pool'
    elif pooling == 'max':
        last_layer = 'max_pool'

    outputs = model.get_layer(name=last_layer).output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def VanTiny(model_name='van_tiny', embed_dims=(32, 64, 160, 256), mlp_ratios=(8, 8, 4, 4), depths=(3, 3, 5, 2),
            weights='imagenet', **kwargs):
    return Van(model_name=model_name, embed_dims=embed_dims, mlp_ratios=mlp_ratios,
               depths=depths, weights=weights, **kwargs)


def VanSmall(model_name='van_small', embed_dims=(64, 128, 320, 512), mlp_ratios=(8, 8, 4, 4), depths=(2, 2, 4, 2),
             weights='imagenet', **kwargs):
    return Van(model_name=model_name, embed_dims=embed_dims, mlp_ratios=mlp_ratios,
               depths=depths, weights=weights, **kwargs)


def VanBase(model_name='van_base', embed_dims=(64, 128, 320, 512), mlp_ratios=(8, 8, 4, 4), depths=(3, 3, 12, 3),
            weights='imagenet', **kwargs):
    return Van(model_name=model_name, embed_dims=embed_dims, mlp_ratios=mlp_ratios,
               depths=depths, weights=weights, **kwargs)


def VanLarge(model_name='van_large', embed_dims=(64, 128, 320, 512), mlp_ratios=(8, 8, 4, 4), depths=(3, 5, 27, 3),
             weights='imagenet', **kwargs):
    return Van(model_name=model_name, embed_dims=embed_dims, mlp_ratios=mlp_ratios,
               depths=depths, weights=weights, **kwargs)
