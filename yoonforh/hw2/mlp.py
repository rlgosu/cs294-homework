import tensorflow as tf
from enum import Enum, IntEnum
import numpy as np
import random
import datetime as dt
import time
import math
import pickle as pk
import os

# scale (1) minmax (2) normal_dist (3) preserve_sign

def scale_minmax(data, minv=None, maxv=None):
    if minv == -1 and maxv == -1 : # no scaling
        return data, -1, -1
    
    if minv is None :
        minv = np.min(data, 0)
    if maxv is None :
        maxv = np.max(data, 0)
    ''' Min Max Normalization

    Parameters
        ----------
        data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
        ----------
        data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
        ----------
        .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - minv
    denominator = maxv - minv
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7), minv, maxv

def descale_minmax(data, minv, maxv):
    if minv is None or maxv is None :
        return data
    if minv == -1 and maxv == -1 : # no scaling
        return data
    
    # noise term prevents the zero division
    return data * (maxv - minv + 1e-7) + minv

def scale_signed(data, minv=None, maxv=None): # value 0 is preserved even after rescale
    if minv == -1 and maxv == -1 : # no scaling
        return data, -1, -1

    if maxv is None :
        maxv = np.max(np.abs(data), 0)
    
    numerator = data
    denominator = maxv
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7), 0.0, maxv

def descale_signed(data, minv, maxv): # value 0 is preserved even after rescale
    if minv is None or maxv is None :
        return data
    if minv == -1 and maxv == -1 : # no scaling
        return data
    
    return data * (maxv + 1e-7)

def scale_zscore(data, mu=None, sigma=None):
    avg = np.mean(data, 0)
    avg = avg if mu is None else avg - mu
    std = np.std(data, 0) if sigma is None else sigma

    numerator = data - avg
    denominator = std + 1e-7
    # noise term prevents the zero division
    return numerator / denominator, avg, std

def descale_zscore(data, mu, std):
    return data * sigma + mu

# util

default_random_seed = 777

NO_RESCALE = { 'minx':-1, 'maxx':-1, 'miny':-1, 'maxy':-1 } 
RESCALE_X = { 'minx':None, 'maxx':None, 'miny':-1, 'maxy':-1 } 
RESCALE_XY = None

TEST_PERCENT = 0.2

def shuffle_XY(X, Y) :
    hstacked = np.hstack((X, Y))
    np.random.shuffle(hstacked)
    _, new_X, new_Y = np.split(hstacked, (0, X.shape[1]), axis=-1)
    return new_X, new_Y

def build_hypothesis(input_placeholder, output_size, scope_name, n_layers, size, activation=tf.tanh, output_activation=None) :
    g = tf.get_default_graph()
    # build the network
    with g.as_default() :
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            neurons = [ size for _ in range(n_layers) ]
            layer = input_placeholder

            for i in range(len(neurons)) :
                neuron = neurons[i]

                layer = tf.layers.dense(layer, neuron,
                                        kernel_initializer = tf.contrib.layers.xavier_initializer(seed=default_random_seed),
                                        activation=activation,
                                        name = 'layer-' + str(i))
            layer = tf.layers.dense(layer, output_size,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(seed=default_random_seed),
                                    activation=output_activation,
                                    name = 'layer-last')
    return layer
