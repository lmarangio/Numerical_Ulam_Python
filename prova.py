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

n = 5
P = np.zeros((n,n))
for i in range(n-1):
	P[i+1,i] = 1
P[0,n-1] = 1
w, v = eig(P, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)
w1 = totuple(w)
e = sorted(w1, reverse = True)[0]
print(e)
t = w1.index(e)
print(v[:,t]/ sum(v[:,t]))