# License: MIT
# Author: Karl Stelzner

import os

import numpy as np
import scipy.misc
import imageio
from observations import mnist


def add_noise(x):
    x = 0.8 * x + 0.1
    x += np.random.normal(0.0, 0.20, size=x.shape)
    x = np.clip(x, 0.0, 1.0)
    return x


def add_structured_noise(images):
    n, height, width = [int(dim) for dim in images.shape[:3]]
    x_offset = np.random.randint(0, 5, n)
    y_offset = np.random.randint(0, 5, n)

    for i in range(n):
        x, y = x_offset[i], y_offset[i]
        while y < height:
            images[i, y] = np.maximum(images[i, y], 0.4)
            y += 5
        while x < width:
            images[i, :, x] = np.maximum(images[i, :, x], 0.4)
            x += 5

    return images


def preprocess(data):
    data = data.astype(np.float32)
    data /= data.max()  # Squash to [0, 1]
    return data


def load_multi_mnist(path, max_digits=2, canvas_size=50, seed=42):
    """
    Code pulled from observations library and customized to
    collect bounding box information.
    Load the multiple MNIST data set [@eslami2016attend]. It modifies
    the original MNIST such that each image contains a number of
    non-overlapping random MNIST digits with equal probability.

    Args:
    path: str.
      Path to directory which either stores file or otherwise file will
      be downloaded and extracted there. Filename is
      `'multi_mnist_{}_{}_{}.npz'.format(max_digits, canvas_size, seed)`.
    max_digits: int, optional.
      Maximum number of non-overlapping MNIST digits per image to
      generate if not cached.
    canvas_size: list of two int, optional.
      Width x height pixel size of generated images if not cached.
    seed: int, optional.
      Random seed to generate the data set from MNIST if not cached.

    Returns:
    Tuple of (np.ndarray of dtype uint8, list)
    `(x_train, y_train), (x_test, y_test)`. Each element in the y's is a
    np.ndarray of labels, one label for each digit in the image.
    """

    def _sample_one(canvas_size, x_data, y_data):
        i = np.random.randint(x_data.shape[0])
        digit = x_data[i]
        label = y_data[i]
        scale = 0.1 * np.random.randn() + 1.3
        resized = scipy.misc.imresize(digit, 1.0 / scale)
        width = resized.shape[0]
        padding = canvas_size - width
        pad_l = np.random.randint(0, padding)
        pad_r = np.random.randint(0, padding)
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, 'constant', constant_values=0)
        bbox = (pad_l, pad_r, pad_l + width, pad_r + width)
        return positioned, label, bbox

    def _sample_multi(num_digits, canvas_size, x_data, y_data):
        canvas = np.zeros((canvas_size, canvas_size))
        labels = []
        bboxes = []
        for _ in range(num_digits):
            positioned_digit, label, bbox = _sample_one(canvas_size, x_data, y_data)
            canvas += positioned_digit
            labels.append(label)
            bboxes.append(bbox)
        labels = np.array(labels, dtype=np.uint8)
        if np.max(canvas) > 255:  # crude check for overlapping digits
            return _sample_multi(num_digits, canvas_size, x_data, y_data)
        else:
            return canvas, labels, bboxes

    def _build_dataset(x_data, y_data, max_digits, canvas_size):
        x = []
        y = []
        data_size = x_data.shape[0]
        data_num_digits = np.random.randint(max_digits + 1, size=data_size)
        x_data = np.reshape(x_data, [data_size, 28, 28])
        bboxes_arr = np.zeros((data_size, max_digits, 4))
        for i, num_digits in enumerate(data_num_digits):
            canvas, labels, bboxes = _sample_multi(num_digits, canvas_size, x_data, y_data)
            x.append(canvas)
            y.append(labels)
            for j, bbox in enumerate(bboxes):
                bboxes_arr[i, j] = bbox
        x = np.array(x, dtype=np.uint8)
        return x, y, bboxes_arr

    path = os.path.expanduser(path)
    cache_filename = 'multi_mnist_{}_{}_{}.npz'.format(
        max_digits, canvas_size, seed)
    if os.path.exists(os.path.join(path, cache_filename)):
        data = np.load(os.path.join(path, cache_filename), allow_pickle=True)
        return (data['x_train'], data['y_train'], data['x_bbox']),\
               (data['x_test'], data['y_test'], data['y_bbox'])

    np.random.seed(seed)
    (x_train, y_train), (x_test, y_test) = mnist(path)
    x_train, y_train, x_bbox = _build_dataset(x_train, y_train, max_digits, canvas_size)
    x_test, y_test, y_bbox = _build_dataset(x_test, y_test, max_digits, canvas_size)
    with open(os.path.join(path, cache_filename), 'wb') as f:
        np.savez_compressed(f, x_train=x_train, y_train=y_train,
                            x_test=x_test, y_test=y_test,
                            x_bbox=x_bbox, y_bbox=y_bbox)
    return (x_train, y_train, x_bbox), (x_test, y_test, y_bbox)


def load_mnist(canvas_size, max_digits=5, path='./data'):
    (x, y, bbox), (x_test, y_test, bbox_test) = \
        load_multi_mnist(path, max_digits=max_digits,
                         canvas_size=canvas_size, seed=42)
    x = preprocess(x)
    x_test = preprocess(x_test)
    # x = 1.0 - x

    # Using FloatTensor to allow comparison with values sampled from Bernoulli.
    counts = np.array([len(objs) for objs in y])
    counts_test = np.array([len(objs) for objs in y_test])
    x = np.expand_dims(x, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x, counts, y, bbox), (x_test, counts_test, y_test, bbox_test)


def load_svhn(path):
    npy_file = open(os.path.join(path, 'svhn_data.npy'), 'rb')
    count_file = open(os.path.join(path, 'svhn_counts.npy'), 'rb')
    obj_file = open(os.path.join(path, 'svhn_objects.npy'), 'rb')
    bg_file = open(os.path.join(path, 'svhn_bg.npy'), 'rb')
    data = np.load(npy_file, allow_pickle=True)
    counts = np.load(count_file, allow_pickle=True)
    objects = np.load(obj_file, allow_pickle=True)
    bgs = np.load(bg_file, allow_pickle=True)
    data = np.expand_dims(data, -1)

    return data, counts, objects, bgs


def add_sprite(canvas):
    while True:
        pos_x = np.random.random_integers(0, 38)
        pos_y = np.random.random_integers(0, 38)

        scale = np.random.random_integers(12, min(min(16, 50 - pos_x), 50 - pos_y))

        cat = np.random.random_integers(0, 2)
        sprite = np.zeros((50, 50, 3))

        if cat == 0:  # draw circle
            center_x = pos_x + scale // 2.0
            center_y = pos_y + scale // 2.0
            for x in range(50):
                for y in range(50):
                    if (x - center_x)**2 + (y - center_y)**2 < (scale // 2.0)**2:
                        sprite[x][y][cat] = 1.0
        elif cat == 1:  # draw square
            sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, cat] = 1.0
        else:  # draw square turned by 45 degrees
            center_x = pos_x + scale // 2.0
            center_y = pos_y + scale // 2.0
            for x in range(50):
                for y in range(50):
                    if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                        sprite[x][y][cat] = 1.0
        mod_canvas = canvas + sprite
        if np.all(np.sum(mod_canvas, axis=2) <= 1.):
            return mod_canvas


def make_sprites(n=50000, path='./data'):
    path = os.path.expanduser(path)
    cache_filename = 'sprites_{}_{}.npz'.format(n, 50)
    if os.path.exists(os.path.join(path, cache_filename)):
        data = np.load(os.path.join(path, cache_filename), allow_pickle=True)
        return (data['x_train'], data['count_train'], None),\
               (data['x_test'], data['count_test'], None)

    images = np.zeros((n, 50, 50, 3))
    counts = np.zeros((n,))
    for i in range(n):
        if i < 100:
            num_sprites = i % 3
        else:
            num_sprites = np.random.random_integers(0, 2)
        counts[i] = num_sprites
        for j in range(num_sprites):
            images[i] = add_sprite(images[i])

    np.clip(images, 0.0, 1.0)
    x_train, count_train = images[:4 * n // 5], counts[:4 * n // 5]
    x_test, count_test = images[4 * n // 5:], counts[4 * n // 5:]

    with open(os.path.join(path, cache_filename), 'wb') as f:
        np.savez_compressed(f, x_train=x_train, count_train=count_train,
                            x_test=x_test, count_test=count_test)

    return (x_train, count_train, None), (x_test, count_test, None)


def load_omniglot(path):
    images = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if len(filename) > 4 and filename[-4:] == '.png':
                fullname = dirname + '/' + filename
                image = imageio.imread(fullname)
                image = scipy.misc.imresize(image, (50, 50))
                images.append(image)

    images = np.stack(images, axis=0)
    images = np.expand_dims(images, -1)
    images = images.astype(np.float32) / 255.0
    np.random.seed(42)
    np.random.shuffle(images)
    print(np.min(images), np.max(images))
    print(images.shape)
    return images
