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

def scale_zscore(data, mu=0, sigma=1): # make N(avg, std) -> N(mu, sigma)
    avg = np.mean(data, 0)
    std = np.std(data, 0)

    # noise term prevents the zero division
    return (data - avg) / (std + 1e-7) * sigma + mu, avg, std

def descale_zscore(data, avg, std, mu=0, sigma=1): # make N(mu, sigma) -> N(avg, std)
    return (data - mu) / (sigma + 1e-7) * std + avg

def check_nan(x) :
    return (x is np.nan or x != x)

def has_nan(x) :
    return sum(check_nan(x)) > 0
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

def build_hypothesis(input_placeholder, output_size, scope_name, n_layers, size,
                     activation=tf.nn.tanh, output_activation=None) :
    print('build_hypothesis(input_placeholder = ', np.shape(input_placeholder), ', output_size = ', output_size,
          ', scope_name = ', scope_name, ', n_layers = ', n_layers, ', size = ', size,
          ', activation = ', activation, ', output_activation = ', output_activation, ')')
    # build the network
    layers = []
    with tf.variable_scope(scope_name) as scope:
        neurons = [ size for _ in range(n_layers) ]
        layer = input_placeholder
        layers.append(layer)

        for i in range(len(neurons)) :
            neuron = neurons[i]

            layer = tf.layers.dense(layer, neuron,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(), # don't fix seed
                                    activation=activation,
                                    name = 'layer-' + str(i))
            layers.append(layer)
        layer = tf.layers.dense(layer, output_size,
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), # don't fix seed
                                activation=output_activation,
                                name = 'layer-last')
        layers.append(layer)
    return layers
