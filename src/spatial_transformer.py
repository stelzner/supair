# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified by Karl Stelzner for the SuPAIR model

import tensorflow as tf
import numpy as np


def scatter_add_tensor(ref, indices, updates, name=None):
    """Add values in updates to ref at indices"""
    ref = tf.convert_to_tensor(ref)
    updates = tf.convert_to_tensor(updates)

    scattered_updates = tf.scatter_nd(indices, updates, ref.shape)
    output = tf.add(ref, scattered_updates)
    return output


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch, height, width, channels = [np.array(dim, dtype=np.int32)
                                                  for dim in im.shape]

            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            x = tf.clip_by_value(x, 0.0, width_f - 1.0001)
            y = tf.clip_by_value(y, 0.0, height_f - 1.0001)

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            base = tf.expand_dims(tf.expand_dims(tf.range(num_batch, dtype=tf.int32), -1), -1)
            base = tf.tile(base, [1, out_width, out_height])

            # TODO: I have no idea why x and y are swapped here...
            idx_a = tf.stack([base, y0, x0], axis=-1)
            idx_b = tf.stack([base, y1, x0], axis=-1)
            idx_c = tf.stack([base, y0, x1], axis=-1)
            idx_d = tf.stack([base, y1, x1], axis=-1)

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            Ia = tf.gather_nd(im, idx_a)
            Ib = tf.gather_nd(im, idx_b)
            Ic = tf.gather_nd(im, idx_c)
            Id = tf.gather_nd(im, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims((x1_f - x) * (y1_f - y), -1)
            wb = tf.expand_dims((x1_f - x) * (y - y0_f), -1)
            wc = tf.expand_dims((x - x0_f) * (y1_f - y), -1)
            wd = tf.expand_dims((x - x0_f) * (y - y0_f), -1)

            source_score = tf.zeros([num_batch, height, width])
            source_score = scatter_add_tensor(source_score, idx_a, (tf.squeeze(wa)))
            source_score = scatter_add_tensor(source_score, idx_b, (tf.squeeze(wb)))
            source_score = scatter_add_tensor(source_score, idx_c, (tf.squeeze(wc)))
            source_score = scatter_add_tensor(source_score, idx_d, (tf.squeeze(wd)))

            source_score = tf.clip_by_value(source_score, 0.0, 1.0)

            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output, source_score

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            x_t, y_t = np.meshgrid(np.linspace(0, width, width),
                                   np.linspace(0, height, height))

            x_t_flat = np.reshape(x_t, (1, -1))
            y_t_flat = np.reshape(y_t, (1, -1))

            ones = np.ones_like(x_t_flat)
            grid = np.vstack([x_t_flat, y_t_flat, ones])
            return tf.constant(grid, dtype=tf.float32)

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = input_dim.shape[0]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.tile(tf.expand_dims(grid, 0), [num_batch, 1, 1])

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            # grid: (num_batch, 3, h*w)
            T_g = tf.matmul(theta, grid)

            x_s = tf.reshape(T_g[:, 0], [num_batch, out_height, out_width])
            y_s = tf.reshape(T_g[:, 1], [num_batch, out_height, out_width])

            return _interpolate(input_dim, x_s, y_s, out_size)

    with tf.variable_scope(name):
        return _transform(theta, U, out_size)


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer

    Parameters
    ----------

    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : tuple of two ints
        the size of the output [out_height,out_width]

    Returns: float
        Tensor of size [num_batch, num_transforms,out_height,out_width,num_channels]
    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i] * num_transforms for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        patches, source_score = transformer(input_repeated, thetas, out_size)
        patch_shape = [num_batch, num_transforms, out_size[0], out_size[1], U.shape[-1]]
        patches = tf.reshape(patches, patch_shape)
        source_score = tf.reshape(source_score, [num_batch, num_transforms] + list(U.shape[1:3]))

        return patches, source_score
