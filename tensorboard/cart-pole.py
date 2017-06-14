import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym

env = gym.make('CartPole-v0')


H_LAYERS_NEURONS = 10
INPUT_DIM = 4
BATCH_SIZE = 5

MAX_EPISODES = 1000000
POINTS_TO_RENDER = 100000
POINTS_TO_SAVE = 2000

gamma = 0.99
running_reward = None
reward_sum = 0
max_reward_avrg = 0


def discount_rewards(r):
	"""
	Takes a 1D float array of rewards and compute discounted reward.
	"""
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


# Creates the Agent
graph = tf.Graph()
with graph.as_default():

		# Input Placeholders
		past_actions = tf.placeholder(shape=[None, 1], dtype=tf.float32)
		past_rewards = tf.placeholder(dtype=tf.float32)
		past_observations = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name='input_x')

		# Layer 1
		weights1 = tf.get_variable("weights1", shape=[INPUT_DIM, H_LAYERS_NEURONS], initializer=tf.contrib.layers.xavier_initializer())		# TODO: Try initializing with "tf.ones"
		hidden1 = tf.nn.relu(tf.matmul(past_observations, weights1))

		# Output Layer
		weights2 = tf.get_variable("weights2", shape=[H_LAYERS_NEURONS, 1], initializer=tf.contrib.layers.xavier_initializer())				# TODO: Try initializing with "tf.ones"
		chosen_action = tf.nn.sigmoid(tf.matmul(hidden1, weights2))

		# Defines our loss
		loglik = tf.log(past_actions*(past_actions - chosen_action) + (1 - past_actions)*(past_actions + chosen_action))
		loss = -tf.reduce_mean(loglik * past_rewards)

		# A way to colect the gradients (Part 1 of "tf.minimize()")
		tvars = tf.trainable_variables()
		gradients = tf.gradients(loss, tvars)

		# A way to update the gradients (Part 2 of "tf.minimize()")
		weights1_gradients = tf.placeholder(dtype=tf.float32)
		weights2_gradients = tf.placeholder(dtype=tf.float32)
		optimizer = tf.train.AdamOptimizer()
		updateGradients = optimizer.apply_gradients(zip([weights1_gradients, weights2_gradients], tvars))


# Runs the session
with tf.Session(graph=graph) as sess:
	
	# Initializes the session
	observations, actions, rewards = [], [], []
	tf.global_variables_initializer().run()												# Inicializes the graph
	rendering = False
	observation = env.reset()															# Starts the environmnet and get initial observation
	saver = tf.train.Saver()															# Creates a saver to save the neural network
	print("Initialized")

	# Asks for restoring previous model
	if raw_input("Would you like to restore the previous model? (y/n) ") == "y":
		saver.restore(sess, "./cart-pole-ckpt/model.ckpt")
		print "Model restored."

	# Clears the collected gradients
	collected_gradients = sess.run(tvars)
	for i, grad in enumerate(collected_gradients):
		collected_gradients[i] = grad * 0

	episode = 1
	while episode <= MAX_EPISODES:

		# Only starts rendering when the agent starts going well
		if reward_sum/BATCH_SIZE > POINTS_TO_RENDER or rendering == True:
			env.render()
			rendering = True

		# Runs the agent to get an action
		observation = np.reshape(observation, [1, INPUT_DIM])							# Reshapes the observation to match neural network input shape
		output = sess.run(chosen_action, feed_dict={past_observations: observation})	# Runs the neural network to get the action
		action = 1 if np.random.uniform() < output else 0								# Adds some noise to the chosen action

		observations.append(observation)												# Records the observations
		actions.append(1 if action==0 else 0)											# Records the actions

		# Act on the environment. Gets the reward and observation
		observation, reward, done, info = env.step(action)
		reward_sum += reward
		rewards.append(reward)															# Records the rewards


		# When the episode ends...
		if done:
			episode += 1
			ep_observations = np.vstack(observations)									# Records episode observations
			ep_actions = np.vstack(actions)												# Records episode actions
			ep_rewards = np.vstack(rewards)												# Records episode rewards
			observations, actions, rewards = [], [], []									# Reinitializes them all

			discounted_ep_rewards = discount_rewards(ep_rewards)						# Gets the time discounted rewards
			discounted_ep_rewards -= np.mean(discounted_ep_rewards)						# Normalizes them
			discounted_ep_rewards /= np.std(discounted_ep_rewards)						# Normalizes them

			# Collects the gradients for this episode
			ep_gradients = sess.run(gradients, feed_dict={past_observations:ep_observations, past_actions:ep_actions, past_rewards:discounted_ep_rewards})
			for i, grad in enumerate(ep_gradients):
				collected_gradients[i] += grad


			# After enough episodes (batch size)...
			if episode % BATCH_SIZE == 0:

				# Updates the neural network with the collected gradients
				sess.run(updateGradients, feed_dict={weights1_gradients:collected_gradients[0], weights2_gradients:collected_gradients[1]})

				# Clears the collected gradients
				for i, grad in enumerate(collected_gradients):
					collected_gradients[i] = grad * 0

				# Saves the best best model so far
				saved = False
				if reward_sum/BATCH_SIZE > (max_reward_avrg)*1.1 and (reward_sum/BATCH_SIZE) > POINTS_TO_SAVE:
					max_reward_avrg = reward_sum/BATCH_SIZE
					save_path = saver.save(sess, "./cart-pole-ckpt/model.ckpt")
					saved = True

				# Give a summary of how well our network is doing for each batch of episodes.
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print 'Episode: %i. Average reward for episode %f.  Total average reward %f.' % (episode, reward_sum/BATCH_SIZE, running_reward/BATCH_SIZE) \
				+ (" *" if saved else "")

				reward_sum = 0
			observation = env.reset()

print episode,'Episodes completed.'