import numpy as np
import tensorflow as tf


S_SIZE = 3
A_SIZE = 4
MAX_EPISODES = 10000
total_reward = np.zeros([S_SIZE, A_SIZE])
e = 0.1 												# Adds noise to the choosing act (random or from the network)


# Defines the bandits
class contextual_bandits():
	def __init__(self):
		self.state = 0
		self.bandits = np.array([[0.2, 0, -0.2, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
		self.num_bandits = self.bandits.shape[0]
		self.num_actions = self.bandits.shape[1]

	def getBandit(self):
		self.state = np.random.randint(0, len(self.bandits))
		return self.state

	def pullArm(self, arm):
		bandit = self.bandits[self.state, arm]
		if np.random.randn(1) > bandit:
			return 1
		else:
			return -1


# Creates the Agent
graph = tf.Graph()
with graph.as_default():

		# Input Placeholders
		past_action = tf.placeholder(shape=[1], dtype=tf.int32)
		past_reward = tf.placeholder(shape=[1], dtype=tf.float32)
		state = tf.placeholder(shape=[1], dtype=tf.int32)

		# Layer 1
		state_OH = tf.contrib.layers.one_hot_encoding(state, S_SIZE)
		weights = tf.Variable(tf.ones([S_SIZE, A_SIZE]))
		output = tf.reshape(tf.nn.sigmoid(tf.matmul(state_OH, weights)), [-1])		# Output comes as "1 x A_SIZE", we want "A_SIZE"

		# Gets choosen action from the output layer
		chosen_action = tf.argmax(output, 0)

		# Training stuff
		responsible_weight = tf.slice(output, past_action, [1])
		loss = -(tf.log(responsible_weight) * past_reward)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		update = optimizer.minimize(loss)


# Runs the session
with tf.Session(graph=graph) as sess:
	
	# Initializes the session
	tf.global_variables_initializer().run()					# Inicializes the graph
	cBandit = contextual_bandits()							# Inicializes the bandits
	print("Initialized")

	for i in xrange(MAX_EPISODES):

		# Gets the state of the environment (which bandit the agent will interact)
		s = cBandit.getBandit()

		# Takes a random action or on from the Neural Network
		if np.random.rand(1) < e:
			action = np.random.randint(A_SIZE)
		else:
			action = sess.run(chosen_action, feed_dict={state:[s]})

		# Gets the reward of the taken action
		reward = cBandit.pullArm(action)

		# Updates the network
		_, ww = sess.run([update, weights], feed_dict={past_reward:[reward], past_action:[action], state:[s]})

		# Updates the scores
		total_reward[s, action] += reward
		if i % 100 == 0:
			print "Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis=1))


# Shows final results
for a in range(cBandit.num_bandits):
    print "The agent thinks arm " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising...."
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print "...and it was right!"
    else:
        print "...and it was wrong!"