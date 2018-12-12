import numpy as np


def split_data(labels_data):
    mask = np.random.rand(len(labels_data)) < 0.7
    train = labels_data[mask]
    test = labels_data[~mask]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test
