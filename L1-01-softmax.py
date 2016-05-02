# Softmax

import numpy as np 
import matplotlib.pyplot as plt 

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])




def softmax(x):
	"""Compute softmax values for x"""
	return np.exp(x) / np.sum(np.exp(x), axis = 0)




def softmax_alt(x):
	"""
	Compute softmax values for x
	This is an alternative solution that works properly
	"""
	x = np.array(x)

	if len(x.shape) >= 2:
		x = (x + 0.0).T
		for i in range(len(x)):

			den = np.sum(np.exp(x[i]))
			for j in range(len(x[0])):
				x[i,j] = np.divide(np.exp(x[i,j]) , den)

		return x.T

	else:
		den = np.sum(np.exp(x) + 0.0)
		for i in range(len(x)):
			x[i] = np.exp(x[i]) / den
		return x




print(softmax(scores))

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth = 2)
plt.show()