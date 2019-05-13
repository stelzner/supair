# License: MIT
# Author: Karl Stelzner

import pickle

import scipy
import visdom
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vis = visdom.Visdom()


def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [1.0, 0.8, 0.7]), -1)


def bounding_box(z_where, x_size):
    z_where = np.reshape(z_where, (2, 3))
    # TODO: support non-quadratic patch-sizes
    corners = np.array([[0.0, 0.0], [0.0, x_size], [x_size, x_size], [x_size, 0.0]])
    corners = np.concatenate((corners, np.ones((4, 1), dtype=np.float32)), axis=1)

    transformed_corners = np.transpose(np.matmul(z_where, np.transpose(corners)))
    return transformed_corners


def colors(k):
    return [(0, 0, 255), (0, 255, 0), (255, 20, 20), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 255, 255)][k]


def img2arr(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size + (3,)).transpose((2, 0, 1))


def draw_overlay(img_arr, text=None, hidden=None):
    img_arr = img_arr.squeeze()
    mode = 'L' if len(img_arr.shape) == 2 else 'RGB'
    img = Image.fromarray((img_arr * 255).astype(np.uint8), mode=mode)
    img = img.convert('RGB')
    if hidden is not None:
        hidden_color = (240, 240, 0)
        for row in range(img.height):
            for col in range(img.width):
                px = img.getpixel((col, row))
                new_px = [int(hidden[row, col] * hidden_color[i] + (1 - hidden[row, col]) * px[i])
                          for i in [0, 1, 2]]
                img.putpixel((col, row), tuple(new_px))
    draw = ImageDraw.Draw(img)
    if text is not None:
        draw.text((0, 0), '{:.2f}'.format(text), fill='red')
    return img2arr(img)


def point_occluded(p, rectangles):
    for r in rectangles:
        if r[0][0] <= p[0] <= r[2][0] and r[0][1] <= p[1] <= r[2][1]:
            return True
    return False


def between(x, a, b):
    if a < b:
        return a < x < b
    else:
        return b < x < a


def find_intersections(l, lines):
    intersections = []
    for line in lines:
        if l[0][0] == l[1][0]:  # l is horizontal
            if (between(l[0][0], line[0][0], line[1][0]) and
                    between(line[0][1], l[0][1], l[1][1])):
                intersections.append([l[0][0], line[0][1]])
        if l[0][1] == l[1][1]:  # l is vertical
            if (between(l[0][1], line[0][1], line[1][1]) and
                    between(line[0][0], l[0][0], l[1][0])):
                intersections.append([line[0][0], l[0][1]])
    return sorted(intersections, key=lambda pt: pt[0] + pt[1])


def draw_rectangles_with_overlap(draw, corners, z_pres):
    """
    Draw rectangles taking overlap into account.
    :param draw: The PIL draw object
    :param corners: 3d np array with shape [num_objects, 4 (corners), 2 (coords)]
    """
    n = corners.shape[0]
    lines_drawn = []
    # rectangles come in in fg to bg order
    for i in range(n):
        color = tuple(map(lambda c: int(c * z_pres[i]), colors(i)))
        for j in range(4):
            line = [corners[i][j], corners[i][(j + 1) % 4]]
            line.sort(key=lambda p: p[0] + p[1])
            waypoints = find_intersections(line, lines_drawn) + [line[1]]
            prev_pt = line[0]
            for pt in waypoints:
                center = [(prev_pt[0] + pt[0]) / 2, (prev_pt[1] + pt[1]) / 2]
                if not point_occluded(center, corners[:i]):
                    lines_drawn.append([prev_pt, pt, color])
                prev_pt = pt
    # draw lines in reverse (i.e., bg to fg) order,
    # so that they occlude each other correctly
    for line in reversed(lines_drawn):
        draw.line([tuple(line[0]), tuple(line[1])], fill=line[2])


def draw_image(img_arr, z_where, z_pres, window_size, text=None):
    mode = 'L' if len(img_arr.shape) == 2 else 'RGB'
    img = Image.fromarray((img_arr * 255).astype(np.uint8), mode=mode)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    rectangles = np.zeros((z_where.shape[0], 4, 2))
    for i in range(z_where.shape[0]):
        if z_pres[i] > 0.3:
            rectangles[i] = bounding_box(z_where[i], window_size)
    if text is not None and text != "":
        draw.text((0, 0), text, fill='red')
    draw_rectangles_with_overlap(draw, rectangles, z_pres)
    return img2arr(img)


def draw_images(img_batch, z_where, z_pres, window_size, text=None):
    result = []
    for i, img in enumerate(img_batch):
        cur_text = None if text is None else text[i]
        result.append(draw_image(img, z_where[i], z_pres[i], window_size, cur_text))
    result = np.stack(result, 0)
    return result


def store_images(images, filename):
    f = open(filename, 'wb')
    results = []
    for i in range(images.shape[0]):
        drawn = draw_overlay(images[i])
        results.append(drawn)
    results = np.stack(results)
    pickle.dump(results, f)
    f.close()


def setup_axis(axis):
    axis.tick_params(axis='both',       # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False,         # ticks along the top edge are off
                     right=False,
                     left=False,
                     labelbottom=False,
                     labelleft=False)   # labels along the bottom edge are off


def show_images(images, nrow=10, text=None, overlay=None, matplot=False):
    images = np.squeeze(images)
    if not matplot:
        images = np.stack([draw_overlay(images[j], None if text is None else text[j])
                           for j in range(len(images))], axis=0)
        vis.images(images, padding=4, nrow=nrow)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
        plt.subplots_adjust(top=1.0, bottom=0.0,
                            left=0.1, right=0.9,
                            wspace=0.1, hspace=-0.15)
        for i, image in enumerate(images):
            cur_axes = axes[i // 4, i % 4]
            setup_axis(cur_axes)
            cur_axes.imshow(image, cmap='gray', interpolation='none')
            if overlay is not None:
                cur_overlay = scipy.misc.imresize(overlay[i], image.shape)
                cur_axes.imshow(cur_overlay, cmap='RdYlGn', alpha=0.5)
        vis.matplot(plt)
        plt.close(fig)


def draw_scene_example(scene, patches, marg_maps, bg_score):
    vis.image(draw_overlay(scene, hidden=1 - bg_score))
    for step in range(patches.shape[0]):
        vis.image(draw_overlay(patches[step], hidden=marg_maps[step]))


class ObjMap:
    def __init__(self, mdl):
        self.mdl = mdl
        self.batch_size = 625
        self.input_ph = tf.placeholder(tf.float32, [self.batch_size, 50, 50, 1])
        self.z_where_ph = tf.placeholder(tf.float32, [self.batch_size, 3, 4])
        self.z_pres_ph = tf.placeholder(tf.float32, [self.batch_size, 3])
        fake_labels = tf.constant(False, dtype=tf.bool, shape=[])

        self.likelihood, _, _, _, _, _ = mdl.likelihood(self.input_ph,
                                                        fake_labels,
                                                        self.z_where_ph,
                                                        self.z_pres_ph)

    def eval(self, images, sess):
        if len(images.shape) < 3:
            images = np.expand_dims(images, -1)
        n = images.shape[0]

        # build z_where grid
        positions_1d = list(range(-7, 43, 2))
        smp = len(positions_1d)
        posx, posy = np.meshgrid(positions_1d, positions_1d)
        posx = np.reshape(posx, (-1, 1))
        posy = np.reshape(posy, (-1, 1))

        z_wheres = np.zeros((smp * smp, 3, 4))

        z_wheres[:, :, 0] = 0.5
        z_wheres[:, :, 1] = 0.6
        z_wheres[:, :, 2] = posx
        z_wheres[:, :, 3] = posy

        z_pres = np.zeros((smp * smp, 3))
        z_pres[:, 0] = 1.

        bl_z_pres = np.zeros((smp * smp, 3))
        bl_images = np.zeros((smp * smp,) + images[0].shape)
        bl_images[:n] = images
        feed_dict = {self.input_ph: bl_images,
                     self.z_pres_ph: bl_z_pres,
                     self.z_where_ph: z_wheres}
        baselines = sess.run(self.likelihood, feed_dict=feed_dict)[:n]
        outputs = []
        for i in range(n):
            image_copies = (np.zeros((smp * smp,) + images[0].shape) +
                            np.expand_dims(images[i], 0))
            ll, marg = sess.run([self.likelihood, self.mdl.marginalize],
                                feed_dict={self.input_ph: image_copies,
                                           self.z_pres_ph: z_pres,
                                           self.z_where_ph: z_wheres})
            marg = np.reshape(marg[0, 0, :28 * 28], (28, 28))
            outputs.append(ll)
        outputs = np.stack(outputs, axis=0)
        outputs -= np.expand_dims(baselines, axis=1)
        outputs = np.reshape(outputs, [n, smp, smp])
        outputs = (outputs / np.max(np.abs(outputs))) * 4
        for i in range(n):
            outputs[i] = np.exp(outputs[i])
            outputs[i] = outputs[i] / (1.0 + outputs[i])

        return outputs


def test_vis():
    img = np.zeros((50, 50), dtype=np.float32)
    z_where = np.zeros((3, 2, 3), dtype=np.float32)
    z_where[:, 0, 0] = 0.5  # x scale
    z_where[:, 1, 1] = 0.5  # y scale
    # positions
    z_where[0, :, 2] = [10, 10]
    z_where[1, :, 2] = [30, 10]
    z_where[2, :, 2] = [20, 15]
    z_pres = [1.0, 1.0, 1.0]
    window_size = 28
    v = draw_image(img, z_where, z_pres, window_size)
    vis.image(v)
    h = np.zeros((50, 50))
    h[:20, :20] = 1.0
    v = draw_overlay(img, hidden=h)
    vis.image(v)


if __name__ == '__main__':
    test_vis()
