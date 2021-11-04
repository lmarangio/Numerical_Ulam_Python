#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import xgboost as xgb
from numpy.linalg import norm
from numpy.linalg import pinv
from math import trunc
from itertools import product
import scipy.sparse as sparse
from scipy.optimize import minimize
from scipy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from boxesfun import* 
from samplefun import*
from comp_invmeas import*
from orbit_generation import*


'''
An algorithm to compute the stationary measure of an approximation of the transfer operator associated to an unknown dynamical system, for which we know an orbit of the system.
'''
N = len(x) 
#Orbit creation
#Arnold's map
#N = 100000
d = 2
delta = 0.1
#f = fA
#Orbit generation 
#o = np.zeros((N,1))
#o[0] = 0.8
#for i in range (1,N):
#	o[i] = f(o[i-1][0])

P = computeUlam(x ,delta, d)
			
y = invmeas(P[0], delta, P[1])
##################### XGB ############################
'''
data = [x[i] for i in range(N-1)]  # 5 entities, each contains 10 features
label = [x[i] for i in range(1,N)]  # binary target
data = np.reshape(data ,[N-1, 1]) 
label = np.reshape(label ,[N-1 ,])
dtrain = xgb.DMatrix(data, label=label)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)
data = np.linspace(0, 1, 100000)
dtest = xgb.DMatrix(data)
ypred = bst.predict(dtest)
plt.plot(data,ypred)
plt.show 
'''
######################################## NN Approach #########################################
#Compute T via MLPRegressor
#Setting up datas for NN
'''
XN= [x[i] for i in range(N-1)]
YN = [x[i] for i in range(1,N)]
XN = np.reshape(XN ,[N-1, 1]) 
YN = np.reshape(YN ,[N-1 ,])
clf = MLPRegressor(alpha=0.00001, hidden_layer_sizes = (100,100), max_iter = 50000, activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
a = clf.fit(XN, YN)

##################First, initial condition##################
#x_ = np.zeros((10000))
#x_[0] = np.random.random_sample()
#for i in range (1,10000):
#  x_[i] = f(x_[i-1])

##################Second, initial condition##################
#x_ = np.linspace(0, 1, 100000)


##################Third, initial condition##################
x_ = np.random.random_sample((100000,))
x_.sort()

pred_x = np.reshape(x_, [100000, 1]) # [160, ] -> [160, 1]
pred_y = clf.predict(pred_x) # predict network output given x_
fx_=[f(x_[i]) for i in range(100000)]
plt.plot(x_, fx_, color = 'r') # plot original function

plt.plot(pred_x, pred_y, '-') # plot network output 
plt.show()

P1 = computeUlampred(pred_x, pred_y,delta,d)
y1 = invmeas(P1[0],delta, P1[1])
print(norm(y-y1))


print(type(XN))
print(type(pred_x))
X1 = np.concatenate((XN, pred_x))
Y1 = np.concatenate((YN, pred_y))

P2 = computeUlampred(X1, Y1,delta,d)
y2 = invmeas(P2[0],delta, P2[1])

print( norm(y-y2))
'''




#######################################################################################################################################################
'''
#Check the theoretical condition

for k in range(N):
	j = getId(o[k], X)
	y = 0
	for t in range(n):
		r = getCenter(t,X)[0]
		y = y + r*P[t,j]
	print(o[k+1] - y)

np.savetxt('matrix.out', P, fmt='%1.4e')

# We compute delta such that a partition of size delta, contains boxes with at most one point inside them.

delta = 100
for i in range(len(o)-1):
  for j in range(i+1,len(o)):
        dist = abs(o[i]-o[j])
        if dist < delta:
            delta = dist
delta /= 2
print(delta)

m = 1 
while 1/m > delta:
	m += 1

delta = 1/m
print(delta)
'''
# Create a partition

'''
# Refinition step : we delete boxes if they not contain a point, and if they are too far from a box which contain a point
blueBoxes = [False]*len(X)
for k in o:
    blueBoxes[getId(k, X)] = True

r = 0.002 #How far we are looking

for i in range(len(X)):
	take = False
	#print(i)
	if blueBoxes[i] == False:
		a = np.array(getCenter(i, X))
		for j in range(len(X)):
			b = np.array(getCenter(j, X))
			if norm(a-b) <= r and blueBoxes[j] == True:
				take = True
				blueBoxes[i] = True
				break
		if take == False:
			blueBoxes[i] = False
X1 = [X[i] for i in range(len(X)) if blueBoxes[i] == True]
X1.sort()
n = len(X1)
print(n)
print(X1[1])

# Create the blu/red boxes list, and the score list 
E1 = np.zeros(n)
score = np.zeros(n)
for k in range(N):
  j = getId(o[k], X1)
  E1[j] = 1
print(E1)
print(sum(E1))
m = 0
while m <= n/2:
	m+=1
print(m)
#We create a first matrix, in which each column correspondent to a blue boxes it is filled with the 'theoretical' values
bnds1 = [(0.001,1) for i in range(m)]
bnds2 = [(0,1) for i in range(m,n)]
bnds = bnds1 + bnds2 
P = np.zeros((n, n))
A = np.zeros((d,n))
for i in range(d):
	for j in range(n):
		A[i][j] = getCenter(j, X1)[i]
print(A)
    
C = np.matrix(np.linalg.pinv(A))
for k in range(N-1):
	j = getId(x[k], X1)
	y = C*np.matrix(x[k+1]).T
	for i in range(n):
		P[i,j]= y[i]


def f(x,k,o,A):
	y = o[k+1]
	for i in range(n):
		y = y - A[0	,i]*x[i]
	y = abs(y)
	return y 
iniz = np.zeros(n)
for k in range(N-1):
	j = getId(o[k], X1)
	print(o[k+1])
	res = minimize(f, iniz, args = (k,o,A) , method='L-BFGS-B', bounds = bnds)
	P[:,j] = res.x
	print(P[:,j]) 
#print(P0)
# Create the blu/red boxes list, and the score list 

full = False
while full == False:
	for j in range(n):
		if E1[j] == 1:
			if E1[j-1] == 0:
				
				for i in range(1,n):
					P[i-1,j-1] = P[i,j]
				P[n-1,j-1] = P[n-1,j]/2
				E1[j-1] = 1
				

			if j != n-1 and E1[j+1] == 0:
				for i in range(n-1):
					P[i+1,j+1] = P[i,j]
				P[0,j+1] = P[0,j]/2
				E1[j+1] = 1
				#print('add')
	for j in range(n-1):
		if E1[j-1] == 1 and E1[j+1] == 1 and E1[j] == 0:
			if P[i,j-1] + P[i,j+1] > 2/3:
				P[i,j] = 1 - P[i,j-1] + P[i,j+1]
			else:
				P[i,j] = P[i,j] = (P[i,j-1] + P[i,j+1])/2
			E1[j] = 1
			#print('add')
	print(sum(E1))
	if sum(E1) == n :
		full = True
		print('finish')
print(E1)
for i in range(n):
	print(sum(P[i,:]))
'''
