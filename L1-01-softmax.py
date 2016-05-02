# Softmax

scores = [3.0, 1.0, 0.2]

import numpy as np 

def softmax(x):
	"""Compute softmax values for x"""
	den = 0
	for i in x:
		den += np.exp(i)
	for i in range(len(x)):
		x[i] = np.exp(x[i]) / den
	return x


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt 
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth = 2)
plt.show()