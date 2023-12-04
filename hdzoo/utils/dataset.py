"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""

import os
import hashlib
import pickle
import numpy as np
import random

from sklearn.preprocessing import Normalizer, MinMaxScaler
from ..utils.logger import log


"""
Loads .choirdat file's information with a cache
returns a tuple: ((samples, labels), features, classes); where features and
classes is a number and samples and labels is a list of vectors

The choir dataset is a set of training/testing datasets developed for
- Kim, Yeseong, Mohsen Imani, Saransh Gupta, Minxuan Zhou, and Tajana S. Rosing. "Massively Parallel Big Data Classification on a Programmable Processing In-Memory Architecture." In 2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1-9. IEEE, 2021.
when I was working at UCSD with Prof. Rosing.
"""
def load_choirdat_with_cache(dataset_path, use_pickle=True):
    def get_file_hash(dataset_path):
        with open(dataset_path, 'rb') as f:
            m = hashlib.sha256()
            m.update(f.read())
        return m.hexdigest()[:16]

    PICKLE_DIR = "HDzoo.cache"
    if use_pickle:
        pickle_filepath = PICKLE_DIR + "/" + get_file_hash(dataset_path)
        if not os.path.exists(PICKLE_DIR):
            os.mkdir(PICKLE_DIR)

        if os.path.exists(pickle_filepath):
            with open(pickle_filepath, 'rb') as f:
                return pickle.load(f)

    def return_with_pickle(*ret):
        if use_pickle:
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(ret, f)
        return ret

    import struct
    with open(dataset_path, 'rb') as f:
        # reads meta information
        features = struct.unpack('i', f.read(4))[0]
        classes = struct.unpack('i', f.read(4))[0]

        # lists containing all samples and labels to be returned
        samples = list()
        labels = list()

        while True:
            # load a new sample
            sample = list()

            # load sample's features
            for i in range(features):
                val = f.read(4)
                if val is None or not len(val):
                    return return_with_pickle(
                            np.array(samples), np.array(labels),
                            features, classes)
                sample.append(struct.unpack('f', val)[0])

            # add the new sample and its label
            label = struct.unpack('i', f.read(4))[0]
            samples.append(sample)
            labels.append(label)

    # Unreachable
    return return_with_pickle(
            np.array(samples, np.float32), np.array(labels, np.int32),
            features, classes)


def load_with_verbose_info(filename):
    x, y, n_features, n_classes = load_choirdat_with_cache(filename)
    log.d("{}\t{} samples\t{} features\t{} classes".format(
        os.path.basename(filename),
        x.shape[0],
        n_features, n_classes))
    return x, y, n_classes


# File Loading
def load_dataset(filename):
    if not filename.endswith('_train.choir_dat'):
         print("Training and testing files are both required", file=os.stderr)
         exit()

    tst_filename = filename.replace('_train.choir_dat', '_test.choir_dat')
    x, y, K = load_with_verbose_info(filename)
    x_test, y_test, K_test = load_with_verbose_info(tst_filename)
    assert K == K_test

    return x, y, x_test, y_test, K


""" Data Normalization Preproecssing """
def normalize(x, x_test=None, normalizer='l2'):
    if normalizer == 'l2':
        scaler = Normalizer(norm='l2').fit(x)
        x_norm = scaler.transform(x)
        if x_test is None:
            return x_norm, None
        else:
            return x_norm, scaler.transform(x_test)
    elif normalizer == 'minmax':
        if x_test is None:
            x_data = x
        else:
            x_data = np.concatenate((x, x_test), axis=0)

        scaler = MinMaxScaler().fit(x_data)
        x_norm = scaler.transform(x)
        if x_test is None:
            return x_norm, None
        else:
            return x_norm, scaler.transform(x_test)

    raise NotImplemented


""" Shuffle dataset """
def shuffle_data(x, y):
    sample_order = list(range(x.shape[0]))
    random.shuffle(sample_order)
    return x[sample_order], y[sample_order]