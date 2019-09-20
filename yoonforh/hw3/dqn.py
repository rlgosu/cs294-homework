import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import mlp

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

# see this for windows support on Atari. https://github.com/openai/gym/issues/11
# I did this one.  (choco install ffmpeg was of no use)
#  => conda install -c conda-forge ffmpeg
#  

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
    self.double_q = double_q

    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for enxt action
    if self.double_q :
      self.act_tp1_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else: # unit8 -> float32
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
    ######

    # YOUR CODE HERE
    # Bellman error : the expectation of the advantage w.r.t. the action a

    # current q network = behavior network (online network)
    self.q_network = q_func(obs_t_float, self.num_actions, 'q_network', False) # q_func returns the q values of each actions. so only discrete actions are applicable

    # next q network = target network.
    # next_q will be the greedy q expectations of s', a' and
    self.target_q_network = q_func(obs_tp1_float, self.num_actions, 'target_q_network', False)

    # r + gamma * Q^*(s', a')
    # self.v_t = self.rew_t_ph + tf.math.reduce_max(tf.where(self.done_mask_ph == 1.0, 0.0, gamma) * self.target_q_network, axis=-1)
    # self.v_t = tf.where(self.done_mask_ph == 1.0, self.rew_t_ph, self.rew_t_ph + gamma * tf.math.reduce_max(self.target_q_network, axis=-1))
    if self.double_q :
      q_tp1 = tf.squeeze(tf.gather(self.target_q_network, tf.expand_dims(self.act_tp1_ph, -1), axis=-1, batch_dims=-1), axis=[-1])
    else :
      q_tp1 = tf.math.reduce_max(self.target_q_network, axis=-1)
      
    self.v_t = self.rew_t_ph + (1.0 - self.done_mask_ph) * gamma * q_tp1
      
    # q network value is expected return of the s, self.act_t_ph. i.e, Q(s, a) --> q_network(obs_t_float, act_t_ph)
    self.q_t = tf.squeeze(tf.gather(self.q_network, tf.expand_dims(self.act_t_ph, -1), axis=-1, batch_dims=-1), axis=[-1])
    self.total_error = tf.reduce_mean(huber_loss(self.v_t - self.q_t)) # calculate huber loss and make them into a single scalar value
    # self.total_error = tf.losses.mean_squared_error(self.v_t, self.q_t)
    # self.total_error = tf.losses.huber_loss(self.v_t, self.q_t)

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_network')
    print('q_func_vars:', q_func_vars)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')
    print('target_q_func_vars:', target_q_func_vars)
    
    ######

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                                      var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####
    # YOUR CODE HERE
    ob = self.last_obs
    saved_idx = self.replay_buffer.store_frame(ob) # returns old idx
    encoded = self.replay_buffer.encode_recent_observation()

    if not self.model_initialized or np.random.rand() <= self.exploration.value(self.t) : # we need to explore by 10 percent
      action = np.random.randint(0, self.num_actions)
      self.rand_count += 1
    else :
      action = np.argmax(self.session.run(self.q_network,
                                          feed_dict= { self.obs_t_ph : np.expand_dims(encoded, 0) } # make 1 batch-size input
      ))
      self.decide_count += 1
      # print('self.obs_t_ph:', np.expand_dims(encoded, 0))
    new_ob, reward, done, _ = self.env.step(action)
    self.replay_buffer.store_effect(saved_idx, action, reward, done)
    
    if done :
      new_ob = self.env.reset()
    self.last_obs = new_ob

  def update_model(self):
    # print('update_model(t:', self.t, ', learning_starts:', self.learning_starts, ', learning_freq:', self.learning_freq,
    #       ', batch_size:', self.batch_size, ', can sample:', self.replay_buffer.can_sample(self.batch_size),')')
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####

      # YOUR CODE HERE
      obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(self.batch_size)
        
      if not self.model_initialized :
        initialize_interdependent_variables(self.session, tf.global_variables(), {
          self.obs_t_ph: obs_t_batch,
          self.obs_tp1_ph: obs_tp1_batch,
        })
        self.session.run(self.update_target_fn)
        self.model_initialized = True

      feed_dict = {
        self.obs_t_ph : obs_t_batch,
        self.act_t_ph : act_t_batch,
        self.rew_t_ph : rew_t_batch,
        self.obs_tp1_ph : obs_tp1_batch,
        self.done_mask_ph : done_mask,
        self.learning_rate : self.optimizer_spec.lr_schedule.value(self.t),
      }
      if self.double_q :
        # apply greedy policy according to the online(behavior) network
        act_tp1_batch = np.argmax(self.session.run(self.q_network,
                                                   feed_dict= { self.obs_t_ph : obs_tp1_batch } # make 1 batch-size input
        ), axis=-1)
        feed_dict[self.act_tp1_ph] = act_tp1_batch
        
      v_t, q_t, total_error = self.session.run([ self.v_t, self.q_t, self.total_error ], feed_dict = feed_dict)
      self.session.run(self.train_fn, feed_dict = feed_dict)

      self.num_param_updates += 1
      if self.num_param_updates % self.target_update_freq == 0 :
        self.session.run(self.update_target_fn)
        
      # if np.sum(np.where(done_mask)) > 0.0 :
      if self.t > 0 and self.t % 10000 == 0 :
        # print('obs_t:', np.array(obs_t_batch), ', act_t:', np.array(act_t_batch))
        print('rew_t:', np.array(rew_t_batch))
        # print('obs_tp1:', np.array(obs_tp1_batch), ', done:', np.array(done_mask))
        print('lr:', self.optimizer_spec.lr_schedule.value(self.t))
        print('v_t:', np.array(v_t))
        print('q_t:', np.array(q_t))
        print('v_t - q_t:', np.array(v_t) - np.array(q_t))
        print('reduce_sum(v_t - q_t):', np.sum(np.array(v_t) - np.array(q_t)))
        print('total_error:', np.array(total_error), ', t:', self.t)

    self.t += 1

  def log_progress(self):
    # print("Timestep %d" % (self.t,),  self.model_initialized)
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      print("exploration count %f, exploitation count %f" %(self.rand_count, self.decide_count))
      self.rand_count = 0
      self.decide_count = 0
      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  alg.rand_count = 0
  alg.decide_count = 0
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

