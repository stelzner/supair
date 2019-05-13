# License: MIT
# Author: Karl Stelzner

import time
import pickle
import os
import copy

import numpy as np
import tensorflow as tf
from scipy.misc import imresize

import config
import model
import rat_spn
import visualize
import datasets
import iou_score


np.set_printoptions(threshold=np.inf)


class SpnReconstructor:
    def __init__(self, spn):
        self.spn = spn
        self.input_ph = tf.placeholder(tf.float32, (1, spn.num_dims))
        self.marginalized = tf.placeholder(tf.float32, (1, spn.num_dims))
        self.spn_out = spn.forward(self.input_ph, self.marginalized)
        self.max_idx_tensors = {}
        for layer in spn.vector_list:
            for vector in layer:
                if isinstance(vector, rat_spn.SumVector):
                    self.max_idx_tensors[vector.name] = vector.max_child_idx

    def reconstruct(self, image, marginalized, sess, sample=False):
        original_shape = image.shape
        image = np.reshape(image, (1, -1))
        marginalized = np.reshape(marginalized, (1, -1))
        feed_dict = {self.input_ph: image, self.marginalized: marginalized}
        max_idxs = sess.run(self.max_idx_tensors, feed_dict=feed_dict)
        recon = self.spn.reconstruct(max_idxs, sess, sample)
        recon = recon * (1 - marginalized)
        recon = np.clip(recon, 0.0, 1.0)
        return np.reshape(recon, original_shape)


class SupairTrainer:
    def __init__(self, conf):
        self.conf = conf

        # determine and create result dir
        i = 1
        log_path = conf.result_path + 'run0'
        while os.path.exists(log_path):
            log_path = '{}run{}'.format(conf.result_path, i)
            i += 1
        os.makedirs(log_path)
        self.log_path = log_path

        if not os.path.exists(conf.checkpoint_dir):
            os.makedirs(conf.checkpoint_dir)

        self.checkpoint_file = os.path.join(self.conf.checkpoint_dir, "model.ckpt")
        input_shape = [conf.batch_size, conf.scene_width, conf.scene_height, conf.channels]
        # build model
        with tf.device(conf.device):
            self.mdl = model.Supair(conf)
            self.in_ph = tf.placeholder(tf.float32, input_shape)
            self.elbo = self.mdl.elbo(self.in_ph)

            self.mdl.num_parameters()

            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(-1 * self.elbo)

        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        if self.conf.load_params:
            self.saver.restore(self.sess, self.checkpoint_file)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

        # load data
        bboxes = None
        if conf.dataset == 'MNIST':
            (x, counts, y, bboxes), (x_test, c_test, _, _) = datasets.load_mnist(
                conf.scene_width, max_digits=2, path=conf.data_path)
            visualize.store_images(x[0:10], log_path + '/img_raw')
            if conf.noise:
                x = datasets.add_noise(x)
                x_test = datasets.add_noise(x_test)
                visualize.store_images(x[0:10], log_path + '/img_noisy')
            if conf.structured_noise:
                x = datasets.add_structured_noise(x)
                x_test = datasets.add_structured_noise(x_test)
                visualize.store_images(x[0:10], log_path + '/img_struc_noisy')
            x_color = np.squeeze(x)
        elif conf.dataset == 'sprites':
            (x_color, counts, _), (x_test, c_test, _) = datasets.make_sprites(50000,
                                                                              path=conf.data_path)
            if conf.noise:
                x_color = datasets.add_noise(x_color)
            x = visualize.rgb2gray(x_color)
            x = np.clip(x, 0.0, 1.0)
            x_test = visualize.rgb2gray(x_test)
            x_test = np.clip(x_test, 0.0, 1.0)
            if conf.noise:
                x = datasets.add_noise(x)
                x_test = datasets.add_noise(x_test)
                x_color = datasets.add_noise(x_color)
        elif conf.dataset == 'omniglot':
            x = 1 - datasets.load_omniglot(path=conf.data_path)
            counts = np.ones(x.shape[0], dtype=np.int32)
            x_color = np.squeeze(x)
        elif conf.dataset == 'svhn':
            x, counts, objects, bgs = datasets.load_svhn(path=conf.data_path)
            self.pretrain(x, objects, bgs)
            x_color = np.squeeze(x)
        else:
            raise ValueError('unknown dataset', conf.dataset)

        self.x, self.x_color, self.counts = x, x_color, counts
        self.x_test, self.c_test = x_test, c_test
        self.bboxes = bboxes

        print('Built model')
        self.obj_reconstructor = SpnReconstructor(self.mdl.obj_spn)
        self.bg_reconstructor = SpnReconstructor(self.mdl.bg_spn)

        tfgraph = tf.get_default_graph()
        self.tensors_of_interest = {
            'z_where': tfgraph.get_tensor_by_name('z_where:0'),
            'z_pres': tfgraph.get_tensor_by_name('z_pres:0'),
            'bg_score': tfgraph.get_tensor_by_name('bg_score:0'),
            'y': tfgraph.get_tensor_by_name('y:0'),
            'obj_vis': tfgraph.get_tensor_by_name('obj_vis:0'),
            'bg_maps': tfgraph.get_tensor_by_name('bg_maps:0')
        }

    def log_and_print_progress(self, elbo, epoch, acc, iou, time_elapsed, log_file):
        print('Epoch {}, elbo {}'.format(epoch, elbo))
        print('Accuracy: {}'.format(acc))

        log_file.write('{}, {}, {}, {}, {}\n'.
                       format(epoch, time_elapsed, acc, iou, elbo))
        log_file.flush()

    def visualize_progress(self, epoch, i, cur_values):
        batch_size = self.conf.batch_size
        start_point = 0  # adjust to show different examples
        img_batch = self.x_color[i * batch_size + start_point: i * batch_size + start_point + 16]
        z_where_batch = cur_values['z_where'][start_point: start_point + 16]
        z_pres_batch = cur_values['z_pres'][start_point: start_point + 16]

        # Show inference results
        results_file = open('{}/img_{}-{}'.format(self.log_path, epoch, i), 'wb')
        drawn_results = visualize.draw_images(img_batch, z_where_batch, z_pres_batch,
                                              self.conf.patch_width)
        visualize.vis.images(drawn_results)
        pickle.dump(drawn_results, results_file)
        results_file.close()

        # Show reconstructions
        recons_file = open('{}/recons_{}-{}'.format(self.log_path, epoch, i), 'wb')
        z_where_batch[:, :, 2] = np.floor(z_where_batch[:, :, 2])
        z_where_batch[:, :, 5] = np.floor(z_where_batch[:, :, 5])
        reconstructions = self.reconstruct_scenes(
            self.x[i * batch_size: i * batch_size + 16],
            cur_values)
        reconstructions = np.squeeze(reconstructions)
        drawn_reconstructions = visualize.draw_images(reconstructions, z_where_batch, z_pres_batch,
                                                      self.conf.patch_width)
        visualize.vis.images(drawn_reconstructions)
        pickle.dump(drawn_reconstructions, recons_file)
        recons_file.close()

    def reconstruct_scenes(self, images, cur_values):
        num_detected = np.sum(np.rint(cur_values['z_pres']), axis=1).astype(np.int32)
        results = []
        for i in range(images.shape[0]):
            n = int(num_detected[i])
            y = cur_values['y'][i]
            z_where = cur_values['z_where'][i]
            obj_vis = cur_values['obj_vis'][i]
            objects = [self.obj_reconstructor.reconstruct(y[k], 1 - obj_vis[k], self.sess)
                       for k in range(n)]
            bg_map = cur_values['bg_maps'][i, n]
            bg = self.bg_reconstructor.reconstruct(images[i], 1 - bg_map, self.sess, sample=True)

            for j in range(n - 1, -1, -1):
                col = int(z_where[j, 2])
                row = int(z_where[j, 5])
                w = int(z_where[j, 0] * self.conf.patch_width)
                h = int(z_where[j, 4] * self.conf.patch_height)
                # check for pathological object dimensions; treat as not present
                if h <= 0 or w <= 0 or row < 0 or col < 0 or row + h > 50 or col + w > 50:
                    continue
                obj = imresize(np.squeeze(objects[j]), (h, w)).astype(np.float32) / 255.0
                bg[row:row + h, col:col + w, 0] = obj

            results.append(bg)

        results = np.stack(results, 0)
        results = np.clip(results, 0.0, 1.0)
        return results

    def pretrain(self, scenes, objects, bg_score, num_examples=100, num_epochs=5000):
        objects = objects[:num_examples]
        scenes = scenes[:num_examples]
        bg_score = bg_score[:num_examples]

        print('pretraining object SPN')
        self.pretrain_spn(self.mdl.obj_spn, objects, num_epochs=num_epochs)
        print('pretraining background SPN')
        self.pretrain_spn(self.mdl.bg_spn, scenes, 1 - bg_score, num_epochs=num_epochs)

    def pretrain_spn(self, spn, inp, marg=None, num_epochs=10):
        n = inp.shape[0]
        batch_size = min(self.conf.batch_size, n)
        inp = np.reshape(inp, (n, -1))
        d = inp.shape[1]
        inp_ph = tf.placeholder(tf.float32, (batch_size, d))
        marg_ph = None
        if marg is not None:
            marg_ph = tf.placeholder(tf.float32, (batch_size, d))
            marg = np.reshape(marg, (n, d))

        spn_out = spn.forward(inp_ph, marg_ph)[:, 0]
        ll = tf.reduce_mean(spn_out)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(-1 * ll)

        batches_per_epoch = n // batch_size
        for epoch in range(num_epochs):
            for i in range(batches_per_epoch):
                batch = inp[i * batch_size: (i + 1) * batch_size]
                feed_dict = {inp_ph: batch}
                if marg is not None:
                    marg_batch = marg[i * batch_size: (i + 1) * batch_size]
                    feed_dict[marg_ph] = marg_batch
                _, cur_ll = self.sess.run([train_op, ll], feed_dict=feed_dict)
            if epoch % 100 == 0:
                print('Pretraining epoch {}, current ll {}'.format(epoch, cur_ll))

    def run_training(self):
        batch_size = self.conf.batch_size
        batches_per_epoch = self.x.shape[0] // batch_size
        sess = self.sess

        perf_log = open(self.conf.log_file, 'a')

        time_elapsed = 0.0
        for epoch in range(self.conf.num_epochs):
            if self.conf.save_params and epoch % 5 == 4:
                self.saver.save(sess, self.checkpoint_file)
            for i in range(batches_per_epoch):
                batch = self.x[i * batch_size: (i + 1) * batch_size]

                start_time = time.time()
                _, cur_elbo, cur_values = sess.run(
                    [self.train_op, self.elbo, self.tensors_of_interest],
                    feed_dict={self.in_ph: batch})
                time_elapsed += time.time() - start_time

                if i % self.conf.log_every == 0:
                    exact_epoch = epoch + i / batches_per_epoch
                    num_detected = np.sum(np.rint(cur_values['z_pres']), axis=1).astype(np.int32)
                    batch_counts = self.counts[i * batch_size: (i + 1) * batch_size]
                    train_acc = np.mean(num_detected == batch_counts)
                    bboxes_pred = iou_score.z_where_to_bboxes(cur_values['z_where'], self.conf)
                    if self.bboxes is not None:
                        iou = iou_score.scene_intersection_over_union(num_detected,
                                                                      batch_counts,
                                                                      bboxes_pred,
                                                                      self.bboxes)
                    else:
                        iou = 0.0

                    if self.conf.get_test_acc:
                        test_acc = self.compute_test_acc()
                        print('test_acc', test_acc)
                        log_acc = test_acc
                    else:
                        log_acc = train_acc

                    self.log_and_print_progress(cur_elbo, exact_epoch, log_acc,
                                                iou, time_elapsed, perf_log)

                    print('avg_obj', np.average(num_detected), np.average(batch_counts))
                    print('train_acc', train_acc)
                    if self.conf.visual:
                        self.visualize_progress(epoch, i, cur_values)

        perf_log.close()


    def compute_test_acc(self):
        batch_size = self.conf.batch_size
        num_batches = self.x_test.shape[0] // batch_size
        z_pres = self.tensors_of_interest['z_pres']
        correct = 0
        for i in range(num_batches):
            x_batch = self.x_test[i * batch_size: (i + 1) * batch_size]
            c_batch = self.c_test[i * batch_size: (i + 1) * batch_size]

            cur_pres = self.sess.run(z_pres, feed_dict={self.in_ph: x_batch})
            num_detected = np.sum(np.rint(cur_pres), axis=1)
            correct += np.sum(num_detected == c_batch)
        test_acc = correct / (num_batches * batch_size)
        return test_acc


def generate_performance_log(conf, log_file='perf_log.csv', num_runs=5):
    if os.path.isfile(log_file):
        os.remove(log_file)
    for i in range(num_runs):
        print('starting run {} for {}...'.format(i, log_file))
        conf.visual = False
        conf.log_file = log_file

        trainer = SupairTrainer(conf)
        trainer.run_training()
        tf.reset_default_graph()


def collect_performance_data_sprites(num_runs, conf, log_file='spair_sprites.csv'):
    print('collection data for sprites data')
    conf = copy.deepcopy(conf)
    conf.bg_max_var = 0.16
    conf.dataset = 'sprites'
    generate_performance_log(conf, log_file=log_file, num_runs=num_runs)


def collect_performance_data_mnist(num_runs, conf, log_file='spair_mnist.csv'):
    print('collecting data for MNIST data')
    conf = copy.deepcopy(conf)
    conf.noise = False
    conf.dataset = 'MNIST'
    generate_performance_log(conf, log_file=log_file, num_runs=num_runs)


def collect_performance_data_noisy_mnist(num_runs, conf, log_file='spair_noisy_mnist.csv'):
    print('collecting data for noisy MNIST data')
    conf = copy.deepcopy(conf)
    conf.bg_max_var = 0.06
    conf.obj_min_var = 0.25
    conf.noise = True
    conf.dataset = 'MNIST'
    generate_performance_log(conf, log_file=log_file, num_runs=num_runs)


def collect_performance_data_struc_mnist(num_runs, conf, log_file='spair_struc_noisy_mnist.csv'):
    print('collecting data for MNIST data with structured noise')
    conf = copy.deepcopy(conf)
    conf.structured_noise = True
    conf.dataset = 'MNIST'
    generate_performance_log(conf, log_file=log_file, num_runs=num_runs)


def run_all_experiments():
    num_runs = 5
    conf = config.SupairConfig()
    conf.num_epochs = 20
    conf.log_every = 50
    conf.get_test_acc = True
    collect_performance_data_mnist(num_runs, conf)
    collect_performance_data_sprites(num_runs, conf)
    collect_performance_data_noisy_mnist(num_runs, conf)
    collect_performance_data_struc_mnist(num_runs, conf)

    conf.background_model = False
    collect_performance_data_mnist(num_runs, conf, log_file='no_bg_mnist.csv')
    collect_performance_data_sprites(num_runs, conf, log_file='no_bg_sprites.csv')
    collect_performance_data_struc_mnist(num_runs, conf, log_file='no_bg_struc.csv')
    collect_performance_data_noisy_mnist(num_runs, conf, log_file='no_bg_noise.csv')


def run_one_experiment():
    conf = config.SupairConfig()
    conf.num_epochs = 20
    conf.log_every = 100
    conf.visual = True
    conf.log_file = 'perf-log.csv'
    conf.dataset = 'MNIST'
    conf.structured_noise = True
    conf.background_model = True
    trainer = SupairTrainer(conf)
    trainer.run_training()
    tf.reset_default_graph()


if __name__ == '__main__':
    run_one_experiment()
    # run_all_experiments()
