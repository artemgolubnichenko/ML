import os

import pandas as pd
import tensorflow as tf

import cnn
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
# path_to_test = os.path.join(configuration.BASE_PATH, 'test')

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

X_train = train.iloc[:, 0].values
for idx, file in enumerate(X_train):
    X_train[idx] = os.path.join(path_to_train, file)
y_train = train.iloc[:, 2:30].values
X_test = test.iloc[:, 0].values
for idx, file in enumerate(X_test):
    X_test[idx] = os.path.join(path_to_train, file)
y_test = test.iloc[:, 2:30].values
log(X_train, y_train)

tf.summary.FileWriterCache.clear()
tf.logging.set_verbosity('DEBUG')
classifier = tf.estimator.Estimator(model_fn=cnn.cnn_model_fn, model_dir=configuration.MODEL_DIR)

train_spec = tf.estimator.TrainSpec(input_fn=lambda: utils.wrap_to_input_fn(X_train, y_train),
                                    max_steps=configuration.MAX_STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: utils.wrap_to_input_fn(X_test, y_test, train=False),
                                  start_delay_secs=configuration.START_DELAY_SECS,
                                  throttle_secs=configuration.THROTTLE_SECS)
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
