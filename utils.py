import numpy as np
import tensorflow as tf
from PIL import Image

import labels


def split_data(labels_data):
    mask = np.random.rand(len(labels_data)) < 0.7
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


def make_rgb_image_from_four_channels(path, image_width=512, image_height=512) -> np.ndarray:
    """
    It makes literally RGB image from source four channels,
    where yellow image will be yellow color, red will be red and so on
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    yellow = np.array(Image.open(path + '_yellow.png'))
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
    image_uint8 = make_rgb_image_from_four_channels(filename)
    return image_uint8, _labels


def wrap_to_input_fn(files, _labels, train=True):
    dataset = tf.data.Dataset.from_tensor_slices((files, _labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(50)
    if train:
        dataset = dataset.repeat().shuffle(100)
    return dataset.make_one_shot_iterator().get_next()
