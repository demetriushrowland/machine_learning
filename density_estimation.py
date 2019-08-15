import matplotlib.pyplot as plt
import numpy as np

def likelihood (data, lam):
	product = 1
	for x in data:
		product *= lam*(np.exp(-lam*x))
	return product

def prior (lam):
	return 1/(2*np.pi)**.5 * np.exp(-.5*lam**2)

def main(lam):
	data = []
	for time in range(100):
		data.append (np.random.exponential(1.0/lam))
	X = [x/100 for x in range (1001)]
	Y = [likelihood (data, lam) * prior (lam) for lam in X]
	plt.plot (X, Y)
	plt.show ()

