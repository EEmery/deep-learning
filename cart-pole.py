import gym
import numpy as np
import tensorflow as tf


# Cares about OpenAI Gym stuff
MAX_EPISODES = 10000
MAX_TIMESTEPS = 100
BATCH_SIZE = 50

env = gym.make("CartPole-v0")		# Sets the simulation to the Cart Pole one


# Creates the neural netwrok to control the cart
graph = tf.Graph()
with graph.as_default():

		# Placeholders
		observations = tf.placeholder(shape=[None, 4], dtype=tf.float32)
		past_actions = tf.placeholder(shape=[None, 1], dtype=tf.float32)
		past_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)

		# Hidden layer 1
		weights1 = tf.Variable(tf.truncated_normal([4, 10]))
		hidden1 = tf.nn.relu(tf.matmul(observations, weights1))

		# Output layer
		weights2 = tf.Variable(tf.truncated_normal([10, 1]))
		actions = tf.nn.sigmoid(tf.matmul(hidden1, weights2))

		# Training stuff
		loss = -tf.reduce_mean(tf.log(past_actions - actions) * past_rewards)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

		# For updating the gradients at the end of every batch
		tvars = tf.trainable_variables()
		newGrads = tf.gradients(loss, tvars)
		W1Grad = tf.placeholder(tf.float32)
		W2Grad = tf.placeholder(tf.float32)
		updateGrads = optimizer.apply_gradients(zip([W1Grad, W2Grad], tvars))


# Changes rewards
def change_rewards(rewards):
	gamma = 0.99
	temp = 0
	for i in reversed(xrange(0, len(rewards))):
		temp = temp*gamma + rewards[i]
		rewards[i] = temp
	rewards -= np.mean(rewards)
	rewards /= np.std(rewards)
	return rewards


# Runs the session
with tf.Session(graph=graph) as sess:
	
	# Initializes the session
	tf.initialize_all_variables().run()
	print("Initialized")

	gradBuffer = sess.run(tvars)
	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	for episode in range(MAX_EPISODES):
		
		observation = env.reset()												# Reset the env on the end of every episode
		episode_observations, episode_actions, episode_rewards = [], [], []		# Cleans our buffers
		
		for step in range(MAX_TIMESTEPS):

			# Here we run the neural network
			if step > 12:
				env.render()													# Renders the frame on every timestep
			observation = np.reshape(observation, [1, 4])						# Reshapes the observation tensor for proper usage
			guess = sess.run([actions], feed_dict={observations:observation})	# Asks the neural net to guess an action
			action = 1 if (guess > 0.5) else 0 									# Generates an action from that guess
			
			# Saves observations and actions pair for latter neural net training
			episode_observations.append(observation)
			episode_actions.append(action)
			
			# Act on env
			observation, reward, done, info = env.step(action)

			# Saves reward from action for latter neural net training
			episode_rewards.append(reward)
			

			# Train the neural net at the end of every episode
			if done:

				# Format observations, actions and rewards for training
				episode_observations = np.vstack(episode_observations)
				episode_actions = np.vstack(episode_actions)
				episode_rewards = np.vstack(change_rewards(episode_rewards))

				# Gets and saves the gradients of this episode
				tGrad = sess.run(newGrads, feed_dict={observations:episode_observations, past_actions:episode_actions, past_rewards:episode_rewards})
				for ix, grad in enumerate(tGrad):
					gradBuffer[ix] += grad

				# Updates the gradients at the end of every batch
				if episode%BATCH_SIZE == 0:
					sess.run(updateGrads, feed_dict={W1Grad:gradBuffer[0], W2Grad:gradBuffer[1]})
					for ix, grad in enumerate(gradBuffer):
						gradBuffer[ix] = grad * 0
					print "Episode " + str(episode) + " endend at step " + str(step)
				break
