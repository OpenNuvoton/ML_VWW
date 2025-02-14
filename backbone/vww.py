"""
VWW class implements a Visual Wake Words (VWW) model using TensorFlow and Keras.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add
from keras import backend


class VWW(object):
    """
    Visual Wake Words (VWW) model class.
    This class implements a VWW model with different configurations.
    """

    def __init__(self, input_shape, vww_type="vww4"):
        if vww_type == "vww4":
            self.vww_model = self.vww4()
        else:
            print("Please choose correct vww type, ex: vww2, vww3, vww4")
            self.vww_model = None

    def vww4(self):
        """
        Builds the VWW4 model architecture.
        Parameters:
        input_shape (tuple): Shape of the input tensor, e.g., (128, 128, 1).
        pooling (str): Type of pooling to be applied at the end of the network. Options are 'avg' for AveragePooling2D and 'max' for MaxPooling2D.
        last_layer (str): Type of the last layer. Options are 'conv' for Conv2D and 'dense' for Dense layer.
        num_classes (int): Number of output classes for the classification task.
        Returns:
        tf.keras.Model: A Keras Model instance.
        """

        channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

        # Define the input layer
        # inputs = Input(shape=input_shape)
        inputs = tf.keras.Input(shape=(128, 128, 1))

        # Initial convolutional block
        x = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name="Conv1")(inputs)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="bn_Conv1")(x)
        x = ReLU(6.0, name="Conv1_relu")(x)

        # vww case is DNAS search for each number of conv's layers
        x = self._inverted_res_block(x, filters=4, stride=1, pointwise=4, block_id=0, shortcut=0)

        x = self._inverted_res_block(x, filters=8, stride=2, pointwise=12, block_id=1, shortcut=0)
        x = self._inverted_res_block(x, filters=14, stride=1, pointwise=16, block_id=2, shortcut=1)

        x = self._inverted_res_block(x, filters=28, stride=2, pointwise=32, block_id=3, shortcut=0)
        x = self._inverted_res_block(x, filters=96, stride=1, pointwise=28, block_id=4, shortcut=1)
        x = self._inverted_res_block(x, filters=96, stride=1, pointwise=28, block_id=5, shortcut=1)

        x = self._inverted_res_block(x, filters=152, stride=2, pointwise=44, block_id=6, shortcut=0)
        x = self._inverted_res_block(x, filters=268, stride=1, pointwise=40, block_id=7, shortcut=1)
        x = self._inverted_res_block(x, filters=268, stride=1, pointwise=52, block_id=8, shortcut=1)
        x = self._inverted_res_block(x, filters=268, stride=1, pointwise=44, block_id=9, shortcut=1)

        x = self._inverted_res_block(x, filters=192, stride=1, pointwise=68, block_id=10, shortcut=0)
        x = self._inverted_res_block(x, filters=172, stride=1, pointwise=68, block_id=11, shortcut=1)
        x = self._inverted_res_block(x, filters=172, stride=1, pointwise=48, block_id=12, shortcut=1)

        x = self._inverted_res_block(x, filters=288, stride=2, pointwise=16, block_id=13, shortcut=0)
        x = self._inverted_res_block(x, filters=96, stride=1, pointwise=16, block_id=14, shortcut=1)
        x = self._inverted_res_block(x, filters=96, stride=1, pointwise=32, block_id=15, shortcut=1)

        x = self._inverted_res_block(x, filters=96, stride=1, pointwise=32, block_id=16, shortcut=0)

        # Last Conv2D
        last_block_filters = 128  # vww case
        x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name="Conv_1")(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1_bn")(x)
        outputs = ReLU(6.0, name="out_relu")(x)

        return tf.keras.Model(inputs, outputs)

    def pad_depth(self, x, desired_channels):
        """
        Pads the input tensor `x` with zeros to match the desired number of channels.
        Args:
            x (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).
            desired_channels (int): The desired number of channels after padding.
        Returns:
            tf.Tensor: A tensor with the same height and width as `x` but with `desired_channels` channels.
        """

        y = backend.zeros_like(x, name="pad_depth1")
        # print(x.shape.as_list()[-1])
        new_channels = desired_channels - x.shape.as_list()[-1]
        y = y[:, :, :, :new_channels]

        return backend.concatenate([x, y])

    def _inverted_res_block(self, inputs, filters, stride, pointwise, block_id, shortcut):
        """
        Builds an inverted residual block with optional shortcut connection.
        Returns:
            tensor: Output tensor of the block.
        """

        channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

        in_channels = backend.int_shape(inputs)[channel_axis]
        # pointwise_conv_filters = int(filters * alpha) # vww case, w/o alpha
        pointwise_conv_filters = filters
        # pointwise_filters = _make_divisible(pointwise_conv_filters, 8) # vww case, w/o alpha
        pointwise_filters = pointwise

        x = inputs
        prefix = f"block_{block_id}_"

        # Expand, Conv2D 1 x 1 x conv_filters
        x = Conv2D(pointwise_conv_filters, kernel_size=1, padding="same", use_bias=False, activation=None, name=prefix + "expand")(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "expand_BN")(x)
        x = ReLU(6.0, name=prefix + "expand_relu")(x)

        # Depthwise, DSCNN2D 3 x 3 x same preivius f
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding="same", name=prefix + "depthwise")(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "depthwise_BN")(x)
        x = ReLU(6.0, name=prefix + "depthwise_relu")(x)

        # Project, Conv2D 1 x 1 x pointwise_filters
        x = Conv2D(pointwise_filters, kernel_size=1, padding="same", use_bias=False, activation=None, name=prefix + "project")(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "project_BN")(x)

        # add shortcut or not
        if shortcut == 1:
            if in_channels < pointwise_filters:

                desired_channels_add = x.shape.as_list()[-1] - in_channels

                # old way
                # pad_inputs = Lambda(self.pad_depth, name='pad_depth', arguments={'desired_channels':desired_channels})(inputs)

                # add the final dim depth
                _pad = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, desired_channels_add)), data_format="channels_first", name=prefix + "pad")(inputs)
                x = Add(name=prefix + "add")([x, _pad])

            elif in_channels > pointwise_filters:

                desired_channels_add = in_channels - x.shape.as_list()[-1]

                # old way
                # pad_inputs = Lambda(self.pad_depth, name='pad_depth', arguments={'desired_channels':desired_channels})(inputs)

                # add the final dim depth
                _pad = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, desired_channels_add)), data_format="channels_first", name=prefix + "pad")(x)
                x = Add(name=prefix + "add")([inputs, _pad])

            else:
                x = Add(name=prefix + "add")([inputs, x])

        return x
