import numpy as np
from statistics import mean
from numpy.linalg import norm

def getCenter(i, X):
    return [mean(X[i][k]) for k in range(len(X[i]))]

def getId(x, X):
	d=1
	board = False
	if np.all([x[i] == 0 for i in range(d)]) == True:
		return 0
	else:
		y = []*d
		z = []*d
		
		for i in range(d):
			if x[i] == 0:
				board = True
				y1 = [i]
				y = y + y1
			else:
				z1 = [i]
				z = z +z1 
			
	if board == False:
		for item,k in enumerate(X):
			if np.all([x[i] > 0 for i in range(d)]) == True  and  np.all([x[i] <= k[i][1] for i in range(d)]) == True:
				return item
	else:
		for item,k in enumerate(X):
			if np.all([x[i] >= k[i][0] for i in y]) == True and np.all([x[i] > 0 for i in z]) == True and np.all([x[i] <= k[i][1] for i in range(d)]) == True:
				return item 

def bdistance(i,j,X):
	a = np.array(getCenter(i, X))
	b = np.array(getCenter(j, X))
	return np.linalg.norm(a-b)

def totuple(a):
	try:
		return tuple(totuple(i) for i in a)
	except TypeError:
		return a