import gym
import tensorflow as tf


# Cares about OpenAI Gym stuff
MAX_EPISODES = 100
MAX_TIMESTEPS = 1000

env = gym.make("CartPole-v0")		# Sets the simulation to the Cart Pole one


# Creates the neural netwrok to control the cart
graph = tf.Graph()
with graph.as_default():

		# Variables
		weights = tf.Variable(tf.truncated_normal([2, 4]))
		bias = tf.Variable(tf.zeros([4]))

		# Placeholders
		acumulated_reward = tf.placeholder(shape=[1], dtype=tf.float32)

		# Training stuff
		loss = -acumulated_reward
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		update = optimizer.minimize(loss)


# Runs the session
with tf.Session(graph=graph) as sess:
	
	# INitializes the session
	tf.initialize_all_variables().run()
	print("Initialized")

	for episode in range(MAX_EPISODES):
		env.reset()								# Reset on the begining of every episode
		for step in range(MAX_TIMESTEPS):
			env.render()						# Renders the frame on every timestep
			osbervation, reward, done, info = env.step(env.action_space.sample())
			if done:
				break

# TODO:
#	- Take arg_max from output as next action
#	- Comunicate session with graph
#	- Run!