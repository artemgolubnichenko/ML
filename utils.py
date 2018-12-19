import numpy as np
import tensorflow as tf

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
    channels = ['red', 'green', 'blue', 'yellow']
    images = []
    for channel in channels:
        image_string = tf.read_file(filename + '_' + channel + '.png')
        image_decoded = tf.image.decode_png(image_string)
        image_converted = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_resized = tf.image.resize_images(image_converted, (image_width, image_height))
        images.append(image_resized)
    image = tf.stack(images, -1)
    image_reshaped = tf.reshape(image, [image_width, image_height, 4])
    return image_reshaped


def _parse_function(filename, _labels):
    image = read_image(filename, 128, 128)
    return image, _labels


def wrap_to_input_fn(files, _labels, train=True):
    dataset = tf.data.Dataset.from_tensor_slices((files, _labels))
    dataset = dataset.map(_parse_function)
    if train:
        dataset = dataset.repeat().shuffle(100)
    dataset = dataset.batch(50)
    return dataset.make_one_shot_iterator().get_next()
