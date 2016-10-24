import gym

MAX_ITERATIONS = 1
MAX_TIMESTEPS = 1000

env = gym.make('CartPole-v0')
for i_iteration in range(MAX_ITERATIONS):
	env.reset()
	for i_step in range(MAX_TIMESTEPS):
		env.render()
		observation, reward, done, info = env.step(env.action_space.sample())
		print observation
		if done:
			print "Episode finished after " + str(i_step) + " time steps"
			break