import numpy as np
import tensorflow as tf


bandits = [0.2, 0, -0.2, -5]
NUM_BANDITS = len(bandits)
MAX_EPISODES = 1000
total_reward = np.zeros(NUM_BANDITS)
e = 0.1 												# Adds noise to the choosing act (random or from the network)


# Creates the bandits
def pullBandit(bandit):
	if np.random.randn(1) > bandit:
		return 1										# Positive reward
	else:
		return -1										# Negative reward


# Creates the Agent
graph = tf.Graph()
with graph.as_default():

		# Placeholders
		past_action = tf.placeholder(shape=[1], dtype=tf.int32)
		past_reward = tf.placeholder(shape=[1], dtype=tf.float32)

		# Layer 1
		weights = tf.Variable(tf.ones([NUM_BANDITS]))
		chosen_action = tf.argmax(weights, 0)

		# Training stuff
		responsible_weight = tf.slice(weights, past_action, [1])
		loss = -(tf.log(responsible_weight) * past_reward)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		update = optimizer.minimize(loss)


# Runs the session
with tf.Session(graph=graph) as sess:
	
	# Initializes the session
	tf.global_variables_initializer().run()
	print("Initialized")

	for i in xrange(MAX_EPISODES):

		# Takes a random action or on from the Neural Network
		if np.random.rand(1) < e:
			action = np.random.randint(NUM_BANDITS)
		else:
			action = sess.run(chosen_action)

		# Gets the reward of the taken action
		reward = pullBandit(bandits[action])

		# Updates the network
		l, _, resp, ww = sess.run([loss, update, responsible_weight, weights], feed_dict={past_reward:[reward], past_action:[action]})

		# Updates the scores
		total_reward[action] += reward
		if i % 50 == 0:
			print "Running reward for the " + str(NUM_BANDITS) + " bandits: " + str(total_reward)


# Shows final results
print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promissing..."
if np.argmax(ww) == np.argmax(-np.array(bandits)):
	print "...and it was right!"
else:
	print "...and it was wrong!"
print "Final weights: " + str(ww)