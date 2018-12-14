import os
import numpy as np
import tensorflow as tf
from PIL import Image

import configuration
import labels


def split_data(labels_data):
    mask = np.random.rand(len(labels_data)) < 0.2
    train = labels_data[mask]
    test = labels_data[~mask]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test


def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = labels.label_names[int(num)]
        row.loc[name] = 1
    return row


def read_image(filename, image_width=512, image_height=512):
    image_string = tf.read_file(filename + '_green.png')
    image_decoded = tf.image.decode_png(image_string)
    image_converted = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_reshaped = tf.reshape(image_converted, [image_width, image_height, 1])
    return image_reshaped


def make_rgb_image_from_four_channels(filename, image_width=512, image_height=512) -> np.ndarray:
    """
    It makes literally RGB image from source four channels,
    where yellow image will be yellow color, red will be red and so on
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    path = os.path.join(configuration.BASE_PATH, filename)
    yellow = np.array(Image.open(path + '_yellow.png'))
    # yellow = read_image(path + '_yellow.png')
    # yellow is red + green
    rgb_image[:, :, 0] += yellow / 2
    rgb_image[:, :, 1] += yellow / 2
    # handling for R,G and B channels
    r_image = Image.open(path + '_red.png')
    rgb_image[:, :, 0] += r_image
    g_image = Image.open(path + '_green.png')
    rgb_image[:, :, 1] += g_image
    b_image = Image.open(path + '_blue.png')
    rgb_image[:, :, 2] += b_image
    # Normalize image
    rgb_image = rgb_image / rgb_image.max() * 255
    return rgb_image.astype(np.uint8)


def _parse_function(filename, _labels):
    # image = make_rgb_image_from_four_channels(filename)
    image = read_image(filename)
    return image, _labels


def wrap_to_input_fn(files, _labels, train=True):
    dataset = tf.data.Dataset.from_tensor_slices((files, _labels))
    dataset = dataset.map(_parse_function)
    if train:
        dataset = dataset.repeat().shuffle(100)
    dataset = dataset.batch(50)
    return dataset.make_one_shot_iterator().get_next()
