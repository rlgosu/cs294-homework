import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1,
                 scope='mbp'):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3
        self._scope = scope

        # print('init dataset state mean:', init_dataset.state_mean, ', state std:', init_dataset.state_std)
        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph = tf.placeholder(shape=[None, self._state_dim], name="state", dtype=tf.float32)
        action_ph = tf.placeholder(shape=[None, self._action_dim], name="action", dtype=tf.float32)
        next_state_ph = tf.placeholder(shape=[None, self._state_dim], name="next_state", dtype=tf.float32)

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        ids = self._init_dataset
        state_norm = utils.normalize(state, ids.state_mean, ids.state_std)
        action_norm = utils.normalize(action, ids.action_mean, ids.action_std)

        self._state_norm = state_norm
        input_norm = tf.concat([state_norm, action_norm], axis=-1)
        self._input_norm = input_norm
        predict_delta_norm = utils.build_mlp(input_norm, self._state_dim, 'model' + self._scope, n_layers=self._nn_layers, reuse=reuse)
        self._predict_norm = state_norm + predict_delta_norm
        next_state_pred_norm = state_norm + predict_delta_norm
        next_state_pred = utils.unnormalize(next_state_pred_norm, ids.state_mean, ids.state_std) # unnormalize the next_state not the delta (delta can be small so the unnormalize can make meaningless errors)
        
        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        ids = self._init_dataset        
        actual_delta_norm = utils.normalize(next_state_ph - state_ph, ids.state_mean, ids.state_std)
        predict_delta_norm = utils.normalize(next_state_pred - state_ph, ids.state_mean, ids.state_std)
        loss = tf.losses.mean_squared_error(actual_delta_norm, predict_delta_norm, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        # state = tf.slice(state_ph, [0, 0], [1, -1])
        state = state_ph[:1, :]
        actions = (self._action_space_high - self._action_space_low) * tf.random.uniform([self._horizon, self._num_random_action_selection, self._action_dim]) \
            + self._action_space_low # (horizon, num_action_selection, action_dim)

        state_tile = tf.tile(state, [self._num_random_action_selection, 1])
        cond = lambda step, x, y : tf.less(step, self._horizon)
        def body(step, curr_states, costs) :
            next_state_preds = self._dynamics_func(curr_states, actions[step, :, :], True) # FIXME. calling _dyamics_func again?
            cost = tf.expand_dims(self._cost_fn(curr_states, actions[step, :, :], next_state_preds), 0)
            return step + 1, next_state_preds, tf.cond(tf.equal(step, 0), lambda: cost, lambda : tf.concat([costs, cost], 0))

        _, _, costs = tf.while_loop(cond, body, [tf.constant(0), state_tile, tf.zeros([0, self._num_random_action_selection])],
                                    shape_invariants=[ tf.TensorShape([]),
                                                       tf.TensorShape([None, self._state_dim]),
                                                       tf.TensorShape([None, self._num_random_action_selection])])
        cost_sums = tf.reduce_sum(costs, axis=0) # (horizon, num_random_action_selection)
        # print('cost_sums:', cost_sums)
        self._cost_sums = cost_sums
        best_index = tf.argmin(cost_sums)
        self._best_index = best_index
        self._state_tile = state_tile
        best_action = actions[0, best_index, :]
        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE

        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, False)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states): # doing mini-batch
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        _, loss = self._sess.run([self._optimizer, self._loss], feed_dict={ 
            self._state_ph : states,
            self._action_ph : actions,
            self._next_state_ph : next_states,
            })

        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        # next_state_pred, state_norm, input_norm, predict_norm = self._sess.run([self._next_state_pred, self._state_norm, self._input_norm, self._predict_norm], feed_dict={ 
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict={ 
            self._state_ph : [state],
            self._action_ph : [action],
            })
        next_state_pred = next_state_pred[0]
        # print('state_norm :', state_norm)
        # print('input_norm :', input_norm)
        # print('predicted_norm :', predict_norm)

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action, cost_sums, best_index, state_tile = self._sess.run([self._best_action, self._cost_sums, self._best_index, self._state_tile], feed_dict={
            self._state_ph : [state],
            })
        # print('best action:', best_action, ', cost sums:', cost_sums, ', best index:', best_index, ', state_tile:', state_tile)

        assert np.shape(best_action) == (self._action_dim,)
        return best_action
