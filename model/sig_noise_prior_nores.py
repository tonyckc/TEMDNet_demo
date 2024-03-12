"""This file is the noise  prior model that conduct preliminary extracting of noise."""
'This model have four dilate convs and four res-blocks '

import tensorflow as tf
import os
from basic_op import conv_op
from basic_op import res_block_layers_v1
from basic_op import res_block_layers_v2
from basic_op import dilated_conv_op


def model(input, reuse=False, name='nosie_prior', training=True, STDDEV=None):
    with tf.variable_scope(name, reuse=reuse):
        dilated_conv1 = dilated_conv_op(input, 'dilated_conv1', 32, training=training, useBN=False, kh=3, kw=3,
                                        rate=1, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv2 = dilated_conv_op(dilated_conv1, 'dilated_conv2', 64, training=training, useBN=False, kh=3, kw=3,
                                        rate=1, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv3 = dilated_conv_op(dilated_conv2, 'dilated_conv3', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv4 = dilated_conv_op(dilated_conv3, 'dilated_conv4', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=3, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv5 = dilated_conv_op(dilated_conv4, 'dilated_conv5', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=4, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv6 = dilated_conv_op(dilated_conv5, 'dilated_conv6', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=4, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv7 = dilated_conv_op(dilated_conv6, 'dilated_conv7', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=3, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)

        dilated_conv8 = dilated_conv_op(dilated_conv7, 'dilated_conv8', 64, training=training, useBN=True, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)

        dilated_conv9 = dilated_conv_op(dilated_conv8, 'dilated_conv9', 32, training=training, useBN=True, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv10 = dilated_conv_op(dilated_conv9, 'dilated_conv10', 1, training=training, useBN=True, kh=3, kw=3,
                                         rate=1, padding="SAME",
                                         activation=None, STDDEV=STDDEV)
        result = dilated_conv10

        return result
