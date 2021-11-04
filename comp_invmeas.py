#Compute and plot stationary measure
from __future__ import division
import numpy as np
from numpy.linalg import norm
from numpy.linalg import pinv
from math import trunc
from itertools import product
import scipy.sparse as sparse
from scipy.optimize import minimize
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.neural_network import MLPRegressor
from boxesfun import* 
from samplefun import*

def computeUlam(o,delta,d):
	y = list(np.arange(0,1+delta,delta))
	Y = [[y[i], y[i+1]] for i in range(len(y)-1)]
	X = list(product(Y, repeat = d))
	X.sort()	
	N = len(o)
	n = len(X)
	print(n)
	print('partition computed')
	P = np.zeros((n,n))
	K = np.zeros(n)
	#for k in range(n):
		#K[k] += 1
	for k in range(N-1):
		j = getId(o[k],X)
		i = getId(o[k+1],X)
		#print(o[k])
		#print(i,j,o[k])
		P[i,j] += 1
	'''	
	for i in range(n):
		for j in range(n):
			if P[i,j] > 0:
				P[i,j] /= K[j]
	'''
	for j in range(n):
		if sum(P[:,j]) != 0:
			P[:,j] /= sum(P[:,j])#K[j]
	for j in range(n):			
		print(sum(P[:,j]))
	print('Ulam matrix builded')
	return (P, X)

def computeUlampred(pred_x, pred_y, delta, d):
	N = len(pred_x)
	pred_x = np.reshape(pred_x, [N,1])
	pred_y = np.reshape(pred_y, [N,1])
	y = list(np.arange(0,1+delta,delta))
	Y = [[y[i], y[i+1]] for i in range(len(y)-1)]
	X = list(product(Y, repeat = d))
	N = len(pred_x)
	n = len(X)
	print('partition computed')
	P = sparse.lil_matrix((n,n))
	K = np.zeros(n)
	for k in range(N-1):
		j = getId(pred_x[k],X)
		i = getId(pred_y[k],X)
		K[j] += 1
		P[i,j] += 1
	for i in range(n):
		for j in range(n):
			if P[i,j] != 0:
				P[i,j] /= K[j]
	print('Ulam matrix builded')
	return (P, X)

def F(x,y,X,v,t):
	r = getId(([x,y]), X)
	return v[r,t]

def invmeas(P, delta, X):
	
	#w,v = sparse.linalg.eigs(P, k = 1, which = 'LM')
	w, v = eig(P, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)
	w1 = totuple(w)
	e = sorted(w1, reverse = True)[0]
	print(e)
	t = w1.index(e)
	v[:,t] /= sum(v[:,t])
	print(sum(v[:,t]))
	#v[:,0] /= sum(v[:,0])
	print(w[t])
	#print(sum(v[:,0]))
	xx = list(getCenter(i,X) for i in range(len(X)))
	v[:,t] /= (delta**2)
	y = v[:,t]
	
	plt.plot(xx,y)
	#plt.savefig('precipitationsBale.png')
	plt.show()
	#######COMMENTS##########
	'''
	I want to take a point (x,y) from the grid (X1, Y1), look at the index of the box in which it belongs, say r, 
	assaign to (x,y) ---> z = v[r,t], where v[:,t] is a vector that I know (the invariant measure)
	Thus I want to plot the graphic of this function...

	'''
	'''
	x = np.linspace(0,1,10)
	y = np.linspace(0,1,10)
	X1, Y1 = np.meshgrid(x,y)
	Z = np.zeros((10,10))
	newv = [v[i,t] for i in range(100)]
	print(Z.shape)
	for i in range(10):
		for j in range(10):
			print(X1[0][i],Y1[j][0])
			r = getId([X1[0][i], Y1[j][0]], X)
			print(r)
			Z[i,j] = newv[r]
	#Z = F(X1,Y1,X,v,t)

	#print(X1.shape)
	#print(Y1.shape)
	#print(Z.shape)
	

	fig = plt.figure()
	#ax = plt.axes(projection='3d')
	#surf = ax.plot_surface(X1, Y1, Z, linewidth=0, antialiased=False)
	plt.show()
	return y
'''

#def FF(x,y):
#	x_=np.zeros((1,2))
#	x_[0][0] = x
#	x_[0][1] = y
	