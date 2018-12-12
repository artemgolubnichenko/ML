import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf

import configuration
import labels
import utils

TRACE = 1


pd.set_option('display.expand_frame_repr', False)


def trace_enabled():
    return TRACE == 1


path_to_train = os.path.join(configuration.BASE_PATH, 'train')
path_to_test = os.path.join(configuration.BASE_PATH, 'test')

path_to_train_csv = os.path.join(configuration.BASE_PATH, 'train.csv')
path_to_test_csv = os.path.join(configuration.BASE_PATH, 'sample_submission.csv')

labels_data = pd.read_csv(path_to_train_csv)
if trace_enabled():
    print(labels_data.shape[0], ' items loaded')
    print(labels_data.head())

y_train, y_test = utils.split_data(labels_data)
if trace_enabled():
    print(y_train.shape[0], ' items in train')
    print(y_train.head())
    print(y_test.shape[0], ' items in test')
    print(y_test.head())


def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = labels.label_names[int(num)]
        row.loc[name] = 1
    return row


for key in labels.label_names.keys():
    y_train[labels.label_names[key]] = 0
    y_test[labels.label_names[key]] = 0


y_train = y_train.apply(fill_targets, axis=1)
y_test = y_test.apply(fill_targets, axis=1)
if trace_enabled():
    print(y_train.head())
    print(y_test.head())


def ohe(x, n):
    return np.eye(n)[x]


def _parse_function(filename, labels):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return tf.reshape(image_float, [512, 512, 1]), labels


def wrap_to_input_fn(files, labels, train=True):
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(50)
    if train:
        dataset = dataset.repeat().shuffle(100)
    return dataset.make_one_shot_iterator().get_next()

# tf_record_pattern = os.path.join(path_to_test, '*_blue.png')
# X_test = np.array(tf.gfile.Glob(tf_record_pattern))
# X_test.sort()

# labels.py = np.loadtxt(os.path.join(path_to_data, 'sample_submission.csv'), np.float32, skiprows=1)
# print(labels.py)
# y_test = np.array([ohe(int(x), 27) for x in labels.py], np.float32)
# print(y_test)

# print(labels.label_names)
