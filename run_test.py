import numpy as np
import rl.core
from matplotlib import pyplot as plt
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, Callback

import gym_pacman  #  in order for env to get registered
from image_utils import center_crop, scale_image
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


# env.seed(1)

class PacmanImageProcessor(rl.core.Processor):

    def process_observation(self, observation):
        image = observation  # just renaming
        image = center_crop(image, [440, 150])
        image = scale_image(image, 0.25)

        # greyscale
        image = image.mean(axis=2)

        # Improve image contrast
        # image[image == color] = 0
        # Next we normalize the image from -1 to +1
        image = (image - 128) / 128 - 1

        return image.transpose()  # bc width and height were swapped

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


env = gym.make('BerkeleyPacmanPO-v0')
nb_actions = env.action_space.n

WINDOW_LENGTH = 4  # number of last frames used

input_shape = (WINDOW_LENGTH, 110, 37)  # 110x37 the size of image, 4 last images are used to get result

IS_TRAINING_MODE = False

# showing what the input image looks like
# plt.imshow(PacmanImageProcessor().process_observation(env.reset()))
# print(PacmanImageProcessor().process_observation(env.reset()).shape)
# plt.show()


def keras_network():
    model = Sequential()

    # (width, height, channels) channels = last WINDOW_LENGTH frames
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    return model


model = keras_network()

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = PacmanImageProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)


dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

class TestLogger2(Callback):
    """ Logger Class for Test """

    def on_train_begin(self, logs={}):
        """ Print logs at beginning of training"""
        print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs={}):
        """ Print logs at end of each episode """
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
        ]
        print(template.format(*variables))

    def on_action_end(self, action, logs={}):
        """ Called at end of each action for each callback in callbackList"""
        print(action)


if IS_TRAINING_MODE:
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_weights.h5f'
    checkpoint_weights_filename = 'dqn_weights_{step}.h5f'
    log_filename = 'dqn_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250)]
    callbacks += [FileLogger(log_filename, interval=100)]

    # dqn.load_weights('dqn_weights_8000.h5f')

    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, action_repetition=1, visualize=False)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False, nb_max_start_steps=0, action_repetition=1, callbacks=[TestLogger2()])
else:  # testing mode!
    weights_filename = 'dqn_weights_96750.h5f'
    #if args.weights:
    #    weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True, nb_max_start_steps=0, action_repetition=1, callbacks=[TestLogger2()])








'''def q_network(X, name_scope):
    # Initialize layers
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
    with tf.compat.v1.variable_scope(name_scope) as scope:
        # initialize the convolutional layers
        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME',
                         weights_initializer=initializer)
        # tf.compat.v1.summary.histogram(‘layer_1’,layer_1)

        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME',
                         weights_initializer=initializer)
        # tf.compat.v1.summary.histogram(‘layer_2’,layer_2)

        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME',
                         weights_initializer=initializer)
        # tf.compat.v1.summary.histogram(‘layer_3’,layer_3)

        flat = flatten(layer_3)

        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)

        # tf.compat.v1.summary.histogram(‘fc’,fc)
        # Add final output layer
        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)

        # tf.compat.v1.summary.histogram(‘output’,output)
        vars = {v.name[len(scope.name):]: v for v in
                tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        # Return both variables and outputs together
        return vars, output


num_episodes = 800
batch_size = 48

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
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)  # Decaying policy with more steps
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action


buffer_len = 20000
# Buffer is made from a deque — double ended queue
exp_buffer = deque(maxlen=buffer_len)


def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0], mem[:, 1], mem[:, 2], mem[:, 3], mem[:, 4]


# we build our Q network, which takes the input X and generates Q values for all the actions in the state

mainQ, mainQ_outputs = q_network(X, 'mainQ')
# similarly we build our target Q network, for policy evaluation
targetQ, targetQ_outputs = q_network(X, 'targetQ')
copy_op = [tf.compat.v1.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

# define a placeholder for our output i.e action
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
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
                'X': [obs],
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
                    'X': o_next_obs, 'in_training_mode': False
                })
                # discounted reward for action: these are our Y-values
                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1 - o_done)
                # merge all summaries and write to the file
                mrg_summary = merge_summary.eval(
                    feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act, in_training_mode: False})
                file_writer.add_summary(mrg_summary, global_step)
                # To calculate the loss, we run the previously defined functions mentioned while feeding inputs
                train_loss, _ = sess.run([loss, training_op],
                                         feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act,
                                                    in_training_mode: True})
                episodic_loss.append(train_loss)
            # after some interval we copy our main Q network weights to target Q network
            if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
            history.append(episodic_reward)
            print('Epochs per episode:', epoch, 'Episode Reward:', episodic_reward, 'Episode number:', len(history))'''
