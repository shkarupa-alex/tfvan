import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfvan.attn import Attention, SpatialAttention


@keras_parameterized.run_all_keras_modes
class TestAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            Attention,
            kwargs={'kernel_size': 13},
            input_shape=[2, 32, 32, 3],
            input_dtype='float32',
            expected_output_shape=[None, 32, 32, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            Attention,
            kwargs={'kernel_size': 21},
            input_shape=[2, 32, 32, 3],
            input_dtype='float32',
            expected_output_shape=[None, 32, 32, 3],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestSpatialAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SpatialAttention,
            kwargs={},
            input_shape=[2, 32, 32, 3],
            input_dtype='float32',
            expected_output_shape=[None, 32, 32, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            SpatialAttention,
            kwargs={},
            input_shape=[2, 32, 32, 3],
            input_dtype='float32',
            expected_output_shape=[None, 32, 32, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
