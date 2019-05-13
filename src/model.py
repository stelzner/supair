# License: MIT
# Author: Karl Stelzner

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.distributions as dists
import numpy as np

import rat_spn
import region_graph
import relu_mlp
import spatial_transformer as stn


class Supair:
    def __init__(self, conf, mean=0.0, seed=None):
        self.conf = conf
        self.lstm_units = conf.lstm_units
        self.lstm = rnn.BasicLSTMCell(self.lstm_units)
        with tf.variable_scope('detection-rnn'):
            self.output_mlp = relu_mlp.ReluMLP(self.lstm_units, [50, 9],
                                               ['s', 'l'], xavier=True)
        self.default_mean = mean

        # build object SPN
        num_dims_patch = conf.patch_height * conf.patch_width * conf.channels
        rg = region_graph.RegionGraph(range(num_dims_patch), seed=seed)
        if conf.random_structure:
            for _ in range(6):
                rg.random_split(2, 2)
        else:
            rg.make_poon_structure(conf.patch_height, conf.patch_width, 10, 2)

        spn_args = rat_spn.SpnArgs()
        spn_args.gauss_min_var = conf.obj_min_var
        spn_args.gauss_max_var = conf.obj_max_var
        self.obj_spn = rat_spn.RatSpn(1, region_graph=rg, args=spn_args,
                                      name='obj-spn', mean=self.default_mean)

        # build bg SPN
        num_dims_bg = conf.scene_height * conf.scene_width * conf.channels
        rg = region_graph.RegionGraph(range(num_dims_bg), seed=seed)
        if conf.random_structure:
            for _ in range(3):
                rg.random_split(2, 1)
        else:
            rg.make_poon_structure(conf.scene_height, conf.scene_width, 20, 2)

        spn_args = rat_spn.SpnArgs()
        spn_args.num_gauss = 6
        spn_args.num_sums = 3
        spn_args.gauss_min_var = conf.bg_min_var
        spn_args.gauss_max_var = conf.bg_max_var
        self.bg_spn = rat_spn.RatSpn(1, region_graph=rg, args=spn_args,
                                     name='bg-spn', mean=self.default_mean)

    def num_parameters(self):
        print('Number of trainable parameters:')
        # rnn
        rnn_params = 0
        lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='detection-rnn')
        for var in lstm_variables:
            cur_params = int(np.prod(var.shape))
            rnn_params += cur_params
        print('rnn', rnn_params)

        spn_params = self.obj_spn.num_params()
        print('obj-spn', spn_params)
        bg_spn_params = self.bg_spn.num_params()
        print('bg-spn', bg_spn_params)
        print('TOTAL', rnn_params + spn_params + bg_spn_params)

    def detect_objects(self, images):
        """
        Use RNN to detect objects in the images.
        :param images: image tensor of shape [num_batch, h, w, c]
        :return: z_pres, a tensor of shape [num_batch, num_steps] with values between 0 and 1,
                 indicating if an object has been found during that step
                 z_where, a tensor of shape [num_batch, num_steps, 4],
                 indicating the position and scale of the objects
                 q_z_where, a tensor of shape [num_batch, num_steps] indicating q(z_where | x)
        """
        batch_size, height, width, channels = [int(dim) for dim in images.shape]

        images_lin = tf.reshape(images, [batch_size, height * width * channels])

        # Initial state of the LSTM memory.
        hidden_state = tf.zeros([batch_size, self.lstm_units])
        current_state = tf.zeros([batch_size, self.lstm_units])
        state = hidden_state, current_state

        z = []

        for _ in range(self.conf.num_steps):
            with tf.variable_scope('detection-rnn'):
                # LSTM variables might get initialized on first call
                hidden, state = self.lstm(images_lin, state)

            z.append(self.output_mlp.forward(hidden))

        z = tf.stack(z, axis=1, name='z')

        z_pres = tf.sigmoid(1 * z[:, :, 0], name='z_pres')
        z_where_mean = tf.sigmoid(z[:, :, 1:5])
        z_where_var = 0.3 * tf.sigmoid(z[:, :, 5:])

        # z_where_mean consists of [sx, sy/sx, x, y]
        # scale sx to [0.3, 0.9], sy to [0.75 * sx, 1.25 * sx],
        # and x, y to [0, 0.9 * canvas_size]
        obj_scale_delta = self.conf.max_obj_scale - self.conf.min_obj_scale
        y_scale_delta = self.conf.max_y_scale - self.conf.min_y_scale
        z_where_mean *= [[[obj_scale_delta, y_scale_delta,
                           0.9 * height, 0.9 * width]]]
        z_where_mean += [[[self.conf.min_obj_scale,
                           self.conf.min_y_scale, 0.0, 0.0]]]
        scale_y = z_where_mean[:, :, 0:1] * z_where_mean[:, :, 1:2]
        z_where_mean = tf.concat((z_where_mean[:, :, 0:1],
                                  scale_y,
                                  z_where_mean[:, :, 2:]), axis=2)

        # sample from variational distribution for z_where
        z_where_dist = dists.MultivariateNormalDiag(loc=z_where_mean, scale_diag=z_where_var)
        z_where = z_where_dist.sample()

        q_z_where = z_where_dist.log_prob(z_where)
        return z_pres, z_where, q_z_where

    def expand_z_where(self, z_where):
        """ turn a batch of z_wheres into affine transformation matrices, i.e.:
            [sx, sy, x, y] -> [[sx, 0, x], [0, sy, y]] -> [sx, 0, x, 0, sy, y]
        """
        # append 0 to get [sx, sy, x, y, 0]
        z_where = tf.concat((z_where, tf.zeros_like(z_where[:, :, 0:1])), axis=2)
        output = tf.gather(z_where, [0, 4, 2, 4, 1, 3], axis=2, name='z_where_exp')
        return output

    def invert_z_where(self, z_where):
        """
        :param z_where: batch of affine transformation matrices with columns [sx, 0, x, 0, sy, y]
        :return: inverse of those matrices, i.e. [1/sx, 0, -x/sx, 0, 1/sy, -y/sy]
        """
        column0 = 1.0 / z_where[..., 0]
        column1 = tf.zeros_like(column0)
        column2 = -1 * z_where[..., 2] * column0
        column4 = 1.0 / z_where[..., 4]
        column5 = -1 * z_where[..., 5] * column4

        result = tf.stack([column0, column1, column2, column1, column4, column5],
                          axis=len(z_where.shape) - 1)
        return result

    def fixed_bg_ll(self, images, background_score):
        """
        Compute likelihood score for the background assuming a fixed normal distribution
        :param images: Scenes of shape (n, h, w, c)
        :param background_score: a tensor of shape [n, h * w] with values between 0 and 1,
                                 determining to what degree a pixel can be considered background
        :return: a likelihood score of shape [batch_size]
        """
        dist = dists.Normal(0.0, 0.35)
        pixel_lls = dist.log_prob(images)
        pixel_lls = tf.reshape(pixel_lls, list(background_score.shape) + [-1])
        pixel_lls = tf.multiply(pixel_lls, tf.expand_dims(background_score, -1))
        # sum accross pixels and batch
        image_lls = tf.reduce_sum(pixel_lls, axis=[1, 2])
        return image_lls

    def background_likelihood(self, images, background_score):
        """
        Compute likelihood score for the background using a dedicated SPN as the bg model
        :param self:
        :param images: Scenes of shape [n, h, w, c]
        :param background_score: a tensor of shape [n, h * w] with values between 0 and 1,
                                 determining to what degree a pixel can be considered background
        :return: a likelihood score of shape [batch_size]
        """

        batch_size, image_h, image_w, image_c = [int(dim) for dim in images.shape]

        # get ll from SPN
        flattened_images = tf.reshape(images, [batch_size, image_h * image_w * image_c])
        spn_outputs = self.bg_spn.forward(flattened_images, 1 - background_score)
        # Get bg_ll from single root node of the SPN
        bg_ll = spn_outputs[:, 0]
        return bg_ll

    def elbo(self, images):
        z_pres, z_where, q_z_where = self.detect_objects(images)

        z_where = self.expand_z_where(z_where)
        z_where = tf.identity(z_where, 'z_where')
        q_z_where_total = tf.reduce_sum(q_z_where, axis=1, keepdims=True)

        p_ys, bg_maps, overlap_ratios = self.obj_ll(images, z_where)
        bg_maps = tf.stack(bg_maps, axis=1)
        flattened_bg_maps = tf.reshape(bg_maps, [self.conf.batch_size,
                                                 self.conf.num_steps + 1,
                                                 -1])
        flattened_bg_maps = tf.identity(flattened_bg_maps, name='bg_maps')

        prev_q_z_pres = 1.0  # probability that all previous objects are present
        overlap_prior = dists.Gamma(1.0, self.conf.overlap_beta)
        overlap_lls = overlap_prior.log_prob(overlap_ratios)

        p_xz_alternatives = []
        q_z_alternatives = []

        for num_obj in range(self.conf.num_steps + 1):
            if self.conf.background_model:
                p_bg = self.background_likelihood(images, flattened_bg_maps[:, num_obj])
            else:
                p_bg = self.fixed_bg_ll(images, flattened_bg_maps[:, num_obj])

            # Sum log-likelihoods of all objects that are present
            p_y = tf.reduce_sum(p_ys[:, :num_obj], axis=1)

            # Prior over number of objects
            p_z_num_obj = tf.log(0.7) - (0.7 * num_obj)
            # Prior over overlap of objects (discouraging excessive overlap)
            p_z_overlap = tf.reduce_sum(overlap_lls[:, :num_obj], axis=1)

            p_xz = p_y + p_bg + p_z_num_obj + p_z_overlap

            # Get probability q(z_pres=num_obj)
            if num_obj < self.conf.num_steps:
                q_z_pres = prev_q_z_pres * (1. - z_pres[:, num_obj])
                prev_q_z_pres *= z_pres[:, num_obj]
            else:
                q_z_pres = prev_q_z_pres

            p_xz_alternatives.append(p_xz)
            q_z_alternatives.append(q_z_pres)

        p_xz_alternatives = tf.stack(p_xz_alternatives, axis=1)
        q_z_alternatives = tf.stack(q_z_alternatives, axis=1)

        # Compute E_q(z_pres | x) [ log p(x, z) ]
        elbo_ll_term = p_xz_alternatives * q_z_alternatives
        elbo_ll_term = tf.reduce_sum(elbo_ll_term, axis=1)

        # Compute E_q(z_pres | x) [ log q(z|x) ]
        eps = 1e-37  # Add epsilon to avoid log(0)
        elbo_qz_term = q_z_alternatives * (tf.log(q_z_alternatives + eps) + q_z_where_total)
        elbo_qz_term = tf.reduce_sum(elbo_qz_term, axis=1)

        elbo = elbo_ll_term - elbo_qz_term
        return tf.reduce_mean(elbo)

    def obj_ll(self, images, z_where):
        num_steps = self.conf.num_steps
        patch_h, patch_w = self.conf.patch_height, self.conf.patch_width
        n, scene_h, scene_w, chans = map(int, images.shape)

        # Extract object patches (also referred to as y)
        patches, object_scores = stn.batch_transformer(images, z_where,
                                                       [patch_h, patch_w])
        patches = tf.identity(patches, name='y')

        # Compute background score iteratively by 'cutting out' each object
        cur_bg_score = tf.ones_like(object_scores[:, 0])
        bg_maps = [cur_bg_score]
        obj_visible = []
        for step in range(num_steps):
            # Everything outside the scene is unobserved -> pad bg_score with zeros
            padded_bg_score = tf.pad(cur_bg_score, [[0, 0], [1, 1], [1, 1]])
            padded_bg_score = tf.expand_dims(padded_bg_score, -1)
            shifted_z_where = z_where[:, step] + [0., 0., 1., 0., 0., 1.]
            vis, _ = stn.transformer(padded_bg_score, shifted_z_where,
                                     [patch_h, patch_w])
            obj_visible.append(vis[..., 0])

            cur_bg_score *= 1 - object_scores[:, step]
            # cur_bg_score = tf.clip_by_value(cur_bg_score, 0.0, 1.0)
            bg_maps.append(cur_bg_score)

        tf.identity(cur_bg_score, name='bg_score')
        obj_visible = tf.stack(obj_visible, axis=1)
        overlap_ratio = 1 - tf.reduce_mean(obj_visible, axis=[2, 3])

        flattened_patches = tf.reshape(patches, [n * num_steps, patch_h * patch_w * chans])
        spn_input = flattened_patches

        pixels_visible = tf.reshape(obj_visible, [n, num_steps, patch_h * patch_w, 1])
        channels_visible = tf.tile(pixels_visible, [1, 1, 1, chans])
        channels_visible = tf.reshape(channels_visible, [n, num_steps, patch_h * patch_w * chans])
        channels_visible = tf.identity(channels_visible, name='obj_vis')
        marginalize = 1 - channels_visible
        marginalize = tf.reshape(marginalize, [n * num_steps, patch_h * patch_w * chans])

        spn_output = self.obj_spn.forward(spn_input, marginalize)
        p_ys = spn_output[:, 0]  # tf.reduce_logsumexp(spn_output + tf.log(0.1), axis=1)
        p_ys = tf.reshape(p_ys, [n, num_steps])
        # Scale by patch size to approximate a calibrated likelihood over x
        p_ys *= z_where[:, :, 0] * z_where[:, :, 4]

        return p_ys, bg_maps, overlap_ratio
