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

class MultiLayerPerceptron(object) :

    default_model_config = dict(neurons = [400, 200],
                                activation = tf.nn.elu, # Using ReLu, which is a discontinuous function, may raise issues. Try using other activation functions, such as tanh or sigmoid.
                                last_activation = None, # final layer activation function. default is no activation
                                optimizer = tf.train.AdadeltaOptimizer, # tf.train.AdamOptimizer, tf.train.ProximalAdagradOptimizer
                                cost_function = tf.losses.mean_squared_error, # tf.losses.huber_loss (robust to outlier)
                                measure_function = 'r_squared', # 'smape' means symmetric_mean_absolute_percentage_error
    )

    default_train_config = dict(start_learning_rate = 0.001,
                                # minimum_learning_rate = 0.000001,
                                num_epochs = 2000,
                                batch_size = 100,
                                keep_prob = 0.5, # for training only (dropout for hidden layer)
                                keep_prob_input = 0.8, # for training only (dropout for input layer)
                                validationset_percent = 0.2, # by default 20 percent is validation set
                                break_accuracy = 0.99, # -1.0, # 0.999, # -1.0
                                early_stopping_epoch_on_max_no_decrease = 500,
                                shuffle_samples_epochs = 200, # shuffle samples per given epochs considering performance. -1 means no shuffling
                                check_accuracy_epochs = 200, # 5000,
                                use_tboard = True,
                                print_cost_interval = 500,
                                print_trained_model = False,
                                )
    
    def __init__(self,
                 X_shape = None, # X shape as list
                 Y_shape = None, # Y shape as list
                 model_config = MultiLayerPerceptron.default_model_config,
                 scope_name = '',
                 restore_mode=False,
                 session=None) :
        self.model_config = model_config
        self.restore_mode = restore_mode
        self.scope_name = scope_name
        if X_shape is not None :
            self.X_shape = list(X_shape)
            self.X_shape[0] = None
        else :
            self.X_shape = None
        if Y_shape is not None :
            self.Y_shape = list(Y_shape)
            self.Y_shape[0] = None
        else :
            self.Y_shape = None

        tf.set_random_seed(MultiLayerPerceptron.default_random_seed)  # reproducibility
        np.random.seed(MultiLayerPerceptron.default_random_seed)

        # Launch new session before graph init
        # interactive session will declare itself as a default session and won't be closed on context destroy (so, should explicity call sess.close()
        if session is None :
            tf.reset_default_graph()
            self.session = tf.InteractiveSession()
        else :
            self.session = session

    def build_hypothesis(self, input_placeholder = None) :
        g = tf.get_default_graph()

        # build the network
        with g.as_default() :
            if input_placeholder is not None :
                self.X = input_placeholder
            else :
                self.X = tf.placeholder(tf.float32, shape=self.X_shape, name='X')
            self.Y = tf.placeholder(tf.float32, shape=self.Y_shape, name='Y')
            self.p_keep_prob = tf.placeholder(tf.float32, name='p_keep_prob')
            self.p_keep_prob_input = tf.placeholder(tf.float32, name='p_keep_prob_input')
            self.p_training = tf.placeholder(tf.bool, name='p_training')
            self.p_lr = tf.placeholder(tf.float32, name='learning_rate')

            with tf.variable_scope(self.scope_name + '-mlp', reuse=tf.AUTO_REUSE) as scope:
                neurons = self.model_config['neurons']
                layer = self.X
                layer = tf.layers.dropout(layer, rate=1-self.p_keep_prob_input, training=self.p_training)
                for i in range(len(neurons)) :
                    neuron = neurons[i]

                    layer = tf.layers.dense(layer, neuron,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(seed=MultiLayerPerceptron.default_random_seed),
                                            activation=self.model_config['activation'],
                                            name = 'layer-' + str(i))
                    layer = tf.layers.dropout(layer, rate=1-self.p_keep_prob, training=self.p_training)
                n_output = self.Y_shape[1]
                layer = tf.layers.dense(layer, n_output,
                                        kernel_initializer = tf.contrib.layers.xavier_initializer(seed=MultiLayerPerceptron.default_random_seed),
                                        activation=self.model_config['last_activation'],
                                        name = 'layer-last')
                    

                self.hypothesis = layer
        return self.hypothesis

    def build_objective(self) :
        cost_fn = self.model_config['cost_function']
        self.cost = cost_fn(self.Y, self.hypothesis)
        tf.summary.scalar("cost", self.cost)
        measure_alg = self.model_config['measure_function']
        if measure_alg == 'r_squared' :
            self.measure = self.r_squared(self.Y, self.hypothesis)
        elif measure_alg == 'smape' :
            self.measure = self.smape(self.Y, self.hypothesis)
        else :
            self.measure = None
        optimizer_fn = self.model_config['optimizer']
        opt = optimizer_fn(learning_rate=self.p_lr)
        self.objective_tensor = opt.minimize(self.cost)

    def initialize_variables(self) :
        with self.session.as_default() :
            # if not self.restore_mode :
            self.session.run(tf.global_variables_initializer())

    def train(self, X, Y, rescale_factor=None, train_config = MultiLayerPerceptron.default_train_config, scale_fn=scale_minmax) :
        learning_rate = train_config['start_learning_rate']
        num_epochs = train_config['num_epochs']
        keep_prob = train_config['keep_prob']
        keep_prob_input = train_config['keep_prob_input']
        batch_size = train_config['batch_size']
        vset_percent = train_config['validationset_percent']
        break_accuracy = train_config['break_accuracy']
        check_accuracy_epochs = train_config['check_accuracy_epochs']
        early_stopping_epoch_on_max_no_decrease = train_config['early_stopping_epoch_on_max_no_decrease']
        print_cost_interval = train_config['print_cost_interval']
        shuffle_samples_epochs = train_config['shuffle_samples_epochs']
        use_tboard = train_config['use_tboard']

        training_costs = np.zeros(num_epochs, dtype=np.float32)
        validation_costs = np.zeros(num_epochs, dtype=np.float32)
        validation_measures = np.zeros(num_epochs, dtype=np.float32)
        min_cost = np.inf
        no_cost_decrease_epochs = 0

        if rescale_factor is None :
            minx, maxx, miny, maxy = None, None, None, None
        else :
            minx, maxx, miny, maxy = rescale_factor['minx'], rescale_factor['maxx'], rescale_factor['miny'], rescale_factor['maxy']
            
        X, minx, maxx = scale_fn(X, minv=minx, maxv=maxx) # rescale X
        Y, miny, maxy = scale_fn(Y, minv=miny, maxv=maxy) # rescale Y
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_output = Y.shape[1]
        n_train = int(n_samples * (1 - vset_percent))
        n_validate = n_samples - n_train

        batch_loop = (n_train - 1) // batch_size + 1

        sess = self.session
        if use_tboard :
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./tboard_logs")
            writer.add_graph(sess.graph)  # Show the graph
        else :
            merged_summary = None

        train_X = X[:n_train]
        train_Y = Y[:n_train]
        validate_X = X[n_train:]
        validate_Y = Y[n_train:]

        if shuffle_samples_epochs > 0 :
            current_XY = np.hstack((X, Y))

        start_time = dt.datetime.now()
        print('Learning starts. It will take some time...', start_time)
        for epoch in range(num_epochs):
            shuffle_samples = shuffle_samples_epochs > 0 and epoch % shuffle_samples_epochs == 0 # shuffle on 0th epoch
            
            if shuffle_samples :
                np.random.shuffle(current_XY) # this will shuffle current_XY in place.
                _, shuffled_X, shuffled_Y = np.split(current_XY, (0, n_features), axis=-1)
                train_X = shuffled_X[:n_train]
                train_Y = shuffled_Y[:n_train]
                validate_X = shuffled_X[n_train:]
                validate_Y = shuffled_Y[n_train:]

            epoch_hyps = np.zeros(Y.shape, dtype=np.float32)
            epoch_costs = np.zeros(batch_loop, dtype=np.float32)

            for m in range(batch_loop) :
                if m == batch_loop - 1 :
                    m_X = train_X[batch_size * m :]
                    m_Y = train_Y[batch_size * m :]
                else :
                    m_X = train_X[batch_size * m : batch_size * (m + 1)]
                    m_Y = train_Y[batch_size * m : batch_size * (m + 1)]

                feed_dict = {self.X:m_X, self.Y:m_Y,
                             self.p_keep_prob:keep_prob,
                             self.p_keep_prob_input:keep_prob_input,
                             self.p_lr:learning_rate,
                             self.p_training:True}
                targets = [ self.hypothesis, self.cost, self.objective_tensor ]
                if use_tboard :
                    targets.append(merged_summary)
                # print('m:', m, ', m_X:', np.shape(m_X), ', m_Y:', np.shape(m_Y), ', feed_dict:', feed_dict)
                results = sess.run(targets, feed_dict = feed_dict)
                if use_tboard :
                    writer.add_summary(results[-1], global_step = epoch * batch_loop + m)

                h_value = results[0]
                epoch_hyps[batch_size * m : batch_size * m + m_Y.shape[0]] = h_value
                cost_value = results[1]
                epoch_costs[m] = cost_value

            training_costs[epoch] = avg_cost = np.mean(epoch_costs)

            validate_feed_dict = {self.X: validate_X, self.Y: validate_Y,
                                  self.p_keep_prob:1.0, self.p_keep_prob_input:1.0, self.p_training:False}
            validate_targets = [ self.hypothesis, self.cost, self.measure ]
            vs_hyps, vs_cost, vs_measure = sess.run(validate_targets, feed_dict=validate_feed_dict)
            validation_costs[epoch] = vs_cost
            validation_measures[epoch] = vs_measure

            if epoch % print_cost_interval == 0 or epoch == num_epochs - 1:
                print('Epoch:', '%04d' % epoch, 'average training cost =', '{:.9f}'.format(avg_cost),
                      'validation cost =', '{:.9f}'.format(vs_cost), 'validation measure =', '{:.9f}'.format(vs_measure), dt.datetime.now())

            if epoch % check_accuracy_epochs == check_accuracy_epochs :
                print('Epoch:', '%04d' % epoch, 'average training cost =', '{:.9f}'.format(avg_cost),
                      'validation cost =', '{:.9f}'.format(vs_cost),
                      'validation measure =', '{:.9f}'.format(vs_measure), dt.datetime.now())

                if break_accuracy > 0 and break_accuracy >= vs_measure :
                    print('Stops the training due to high validation measure', vs_measure, ' exceeded the criteria', break_accuracy)
                    training_costs = training_costs[:epoch + 1] # strip un-run epochs
                    validation_costs = validation_costs[:epoch + 1] # strip un-run epochs
                    validation_measures = validation_measures[:epoch + 1] # strip un-run epochs
                    break

            if early_stopping_epoch_on_max_no_decrease > 0 :
                if vs_cost < min_cost :
                    min_cost = vs_cost
                    no_cost_decrease_epochs = 0
                else :
                    no_cost_decrease_epochs = no_cost_decrease_epochs + 1
                    if no_cost_decrease_epochs >= early_stopping_epoch_on_max_no_decrease :
                        print('Epoch:', '%04d' % epoch, 'average training cost =', '{:.9f}'.format(avg_cost),
                              'validation cost =', '{:.9f}'.format(vs_cost),
                              'validation measure =', '{:.9f}'.format(vs_measure), dt.datetime.now())
                        # FIXME : in reality, i need to restore variables saved when it was not decreasing but i do not. maybe in the future ..
                        print('Stops the training since cost is not reduced during ', no_cost_decrease_epochs, ' epochs.')
                        training_costs = training_costs[:epoch + 1] # strip un-run epochs
                        validation_costs = validation_costs[:epoch + 1] # strip un-run epochs
                        validation_measures = validation_measures[:epoch + 1] # strip un-run epochs
                        break

        end_time = dt.datetime.now()
        print('Training(learning) Finished!', end_time)
        print('Training took ', '%10d' % ((end_time - start_time).total_seconds()),
              ' seconds.')
   
        rescale_factor = { 'minx':minx, 'maxx':maxx, 'miny':miny, 'maxy':maxy }
        return training_costs, validation_costs, validation_measures, rescale_factor               

    def test(self, X, Y, rescale_factor=None, scale_fn=scale_minmax, descale_fn=descale_minmax) :
        start_time = dt.datetime.now()
        g = tf.get_default_graph()

        if rescale_factor is not None :
            X, _, _ = scale_fn(X, rescale_factor['minx'], rescale_factor['maxx'])
            Y, _, _ = scale_fn(Y, rescale_factor['miny'], rescale_factor['maxy'])
            
        with g.as_default() :
            hyps, cost, measure = self._test_model(X, Y)
            if rescale_factor is not None :
                hyps = descale_fn(hyps, rescale_factor['miny'], rescale_factor['maxy'])
            end_time = dt.datetime.now()
            print('Prediction took ', '%10d' % ((end_time - start_time).total_seconds()),
                  ' seconds.')
            print('Started at ', start_time, ' and finished at ', end_time)
            return hyps, cost, measure

    def _test_model(self, X, Y) :
        test_feed_dict = {self.X: X, self.Y: Y,
                          self.p_keep_prob:1.0, self.p_keep_prob_input:1.0, self.p_training:False}
        test_targets = [ self.hypothesis, self.cost, self.measure ]

        sess = self.session
        hyps, cost, measure = sess.run(test_targets, feed_dict=test_feed_dict)
        return hyps, cost, measure

    def infer(self, X, rescale_factor=None, scale_fn=scale_minmax, descale_fn=descale_minmax) :
        g = tf.get_default_graph()

        if rescale_factor is not None :
            X, _, _ = scale_fn(X, rescale_factor['minx'], rescale_factor['maxx'])
        
        with g.as_default() :
            hyps = self._infer_model(X)
            if rescale_factor is not None :
                hyps = descale_fn(hyps, rescale_factor['miny'], rescale_factor['maxy'])
                
            return hyps

    def _infer_model(self, X) :
        test_feed_dict = {self.X: X,
                          self.p_keep_prob:1.0, self.p_keep_prob_input:1.0, self.p_training:False}
        test_targets = [ self.hypothesis ]

        sess = self.session
        hyps = sess.run(test_targets, feed_dict=test_feed_dict)
        return hyps
    
    def r_squared(self, y, h) :
        # in tf.reduce_mean, if axis has no entries, all dimensions are reduced, and a tensor with a single element is returned
        total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y, 0))))  # reduce_mean by 0-axis maintains vector dimension
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, h)))
        r_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))
        return r_squared

    def smape(self, y, h) :
        return tf.reduce_mean(2.0 * tf.abs(tf.subtract(y, h)) / tf.maximum(1e-7, (tf.abs(y) + tf.abs(h)))) # tf.maximum is used to avoid nan
        
    def check_nan(self, value) :
        return value is None or math.isnan(value)

    def save_model(self, save_file_name) :
        # self._dump_graph('save_model(' + save_file_name + ')')
        
        tf.train.Saver().save(self.session, save_file_name)

    def save_scale(self, save_model_name, rescale_factor) :
        with open(save_model_name + '.scale.pkl', 'wb') as f :
            pk.dump(rescale_factor, f, pk.HIGHEST_PROTOCOL)
        
    def read_scale(self, save_model_name) :
        with open(save_model_name + '.scale.pkl', 'rb') as f :
            rescale_factor = pk.load(f)
            return rescale_factor
        
    def _dump_graph(self, where) :
        print('')

        print('--- dumping tensorflow graph [', where, '] ---')
        g = tf.get_default_graph()
        print('default tf graph :', g)

        # debug graphs
        keys = g.get_all_collection_keys()
        print('current name scope :', g.get_name_scope())
        for key in keys :
            print('all graph (', key, ')  :', g.get_collection(key))
        print('') 
        print('')

       
    def restore_model(self, saved_dir) :
        print('saved dir:', saved_dir)

        with self.session.as_default() :
            # self._dump_graph('restore_model(' + saved_dir + ')')
            
            reader = tf.train.NewCheckpointReader(saved_dir)
            # for var_name in reader.get_variable_to_shape_map() :
            #     print(var_name)
        
            tf.train.Saver().restore(self.session, saved_dir)
