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


def log(*args):
    if TRACE == 1:
        for arg in args:
            print(arg)


path_to_train = os.path.join(configuration.BASE_PATH, 'train')
path_to_test = os.path.join(configuration.BASE_PATH, 'test')

path_to_train_csv = os.path.join(configuration.BASE_PATH, 'train.csv')
path_to_test_csv = os.path.join(configuration.BASE_PATH, 'sample_submission.csv')

labels_data = pd.read_csv(path_to_train_csv)
log(labels_data.shape[0], labels_data.head())

train, test = utils.split_data(labels_data)
log(train.shape[0], train.head(), test.shape[0], test.head())

for key in labels.label_names.keys():
    train[labels.label_names[key]] = 0
    test[labels.label_names[key]] = 0

train = train.apply(utils.fill_targets, axis=1)
test = test.apply(utils.fill_targets, axis=1)
log(train.head(), test.head())

X_train = train.iloc[:, 0]
y_train = train.iloc[:, 2:30]
X_test = test.iloc[:, 0]
y_test = test.iloc[:, 2:30]
log(X_train.head(), y_train.head())


# def load_image(self, path):
#     R = Image.open(path + '_red.png')
#     G = Image.open(path + '_green.png')
#     B = Image.open(path + '_blue.png')
#     Y = Image.open(path + '_yellow.png')
#
#     im = np.stack((
#         np.array(R),
#         np.array(G),
#         np.array(B),
#         np.array(Y)), -1)
#
#     im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
#     im = np.divide(im, 255)
#     return im