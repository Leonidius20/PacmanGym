from PIL import Image
import gym
from keras.layers import Conv2D, Flatten, Dense

import gym_pacman
import time
import cv2
from image_utils import center_crop, scale_image
import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf
from tf_slim.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import random
from datetime import datetime

env = gym.make('BerkeleyPacmanPO-v0')

# env.seed(1)




def preprocess_observation(image):
  # Crop and resize the image
  image = center_crop(image, [440, 150])
  image = scale_image(image, 0.25)

  # greyscale
  image = image.mean(axis=2)

  # Improve image contrast
  # image[image == color] = 0
  # Next we normalize the image from -1 to +1
  image = (image - 128) / 128 - 1

  print(image.shape[1], image.shape[0])

  return image



n_outputs = env.action_space.n


def q_network(X, name_scope):
# Initialize layers
  initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
  with tf.compat.v1.variable_scope(name_scope) as scope:

    # initialize the convolutional layers
    layer_1 = conv2d(X, num_outputs=32, kernel_size=(8,8), stride=4, padding='SAME', weights_initializer=initializer)
    # tf.compat.v1.summary.histogram(‘layer_1’,layer_1)

    layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4,4),    stride=2, padding='SAME', weights_initializer=initializer)
    # tf.compat.v1.summary.histogram(‘layer_2’,layer_2)

    layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3,3), stride=1, padding='SAME', weights_initializer=initializer)
    # tf.compat.v1.summary.histogram(‘layer_3’,layer_3)

    flat = flatten(layer_3)

    fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)

    # tf.compat.v1.summary.histogram(‘fc’,fc)
    #Add final output layer
    output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)

    # tf.compat.v1.summary.histogram(‘output’,output)
    vars = {v.name[len(scope.name):]: v for v in tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
    #Return both variables and outputs together
    return vars, output



num_episodes = 800
batch_size = 48
input_shape = (None, 110, 37, 1)  # 110x37 the size of image

learning_rate = 0.001
X_shape = (None, 88, 80, 1)
discount_factor = 0.97
global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000





epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000
#
def epsilon_greedy(action, step):
  # p = np.random.random(1).squeeze() #1D entries returned using squeeze
  epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps) #Decaying policy with more steps
  if np.random.rand() < epsilon:
    return np.random.randint(n_outputs)
  else:
    return action



buffer_len = 20000
#Buffer is made from a deque — double ended queue
exp_buffer = deque(maxlen=buffer_len)
def sample_memories(batch_size):
  perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
  mem = np.array(exp_buffer)[perm_batch]
  return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]




# we build our Q network, which takes the input X and generates Q values for all the actions in the state

mainQ, mainQ_outputs = q_network(X, 'mainQ')
# similarly we build our target Q network, for policy evaluation
targetQ, targetQ_outputs = q_network(X, 'targetQ')
copy_op = [tf.compat.v1.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)




# define a placeholder for our output i.e action
y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))
# now we calculate the loss which is the difference between actual value and predicted value
loss = tf.reduce_mean(input_tensor=tf.square(y - Q_action))
# we use adam optimizer for minimizing the loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.compat.v1.global_variables_initializer()
loss_summary = tf.compat.v1.summary.scalar('LOSS', loss)
merge_summary = tf.compat.v1.summary.merge_all()
file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())



with tf.compat.v1.Session() as sess:
  init.run()
  # for each episode
  history = []
  for i in range(num_episodes):
    done = False
    obs = env.reset()
    epoch = 0
    episodic_reward = 0
    actions_counter = Counter()
    episodic_loss = []
    # while the state is not the terminal state
    while not done:
      # get the preprocessed game screen
      obs = preprocess_observation(obs)
      # feed the game screen and get the Q values for each action,
      actions = mainQ_outputs.eval(feed_dict={
        'X':[obs],
        'in_training_mode': False}
      )
      # get the action
      action = np.argmax(actions, axis=-1)
      actions_counter[str(action)] += 1
      # select the action using epsilon greedy policy
      action = epsilon_greedy(action, global_step)
      # now perform the action and move to the next state, next_obs, receive reward
      next_obs, reward, done, _ = env.step(action)
      # Store this transition as an experience in the replay buffer! Quite important
      exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])
      # After certain steps we move on to generating y-values for Q network with samples from the experience replay buffer
      if global_step % steps_train == 0 and global_step > start_steps:
          o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
          # states
          o_obs = [x for x in o_obs]
          # next states
          o_next_obs = [x for x in o_next_obs]
          # next actions
          next_act = mainQ_outputs.eval(feed_dict={
            'X':o_next_obs, 'in_training_mode':False
          })
          #discounted reward for action: these are our Y-values
          y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done)
          # merge all summaries and write to the file
          mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:False})
          file_writer.add_summary(mrg_summary, global_step)
          # To calculate the loss, we run the previously defined functions mentioned while feeding inputs
          train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
          episodic_loss.append(train_loss)
      # after some interval we copy our main Q network weights to target Q network
      if (global_step+1) % copy_steps == 0 and global_step > start_steps:
        copy_target_to_main.run()
      obs = next_obs
      epoch += 1
      global_step += 1
      episodic_reward += reward
      history.append(episodic_reward)
      print('Epochs per episode:', epoch, 'Episode Reward:', episodic_reward,'Episode number:', len(history))