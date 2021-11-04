from __future__ import division
import numpy as np
from math import trunc

#Tsuda map
def fT(x):
	y = x + 0.1 + 0.08*np.sin(4*np.pi*x)
	if y > 1:
		y = y - trunc(y)
	return y
#Arnold's map
def fA(x):
  y = x + 0.4 - (0.7/(2*np.pi))*np.sin(2*np.pi*x)
  if y > 1:
    y = y - trunc(y)
  
  return y
#Tent map r = 0
def ftent(x):
	if x <=	1/2:
		y = 2*x
	else :
		y = 2*(1-x)
	return y
#Boole map (di solito é definita su R...)
def fB(x):
	y = x - 1/x
	return y

def f1(x):
	y = (x+0.5)**2 
	if y>1:
		y = y - trunc(y)
	return y 

def f2(x):
	y = np.sin(x)
	if y > 1:
		y = y - trunc(y) 
	return y 
# F
def f3(x):
	y = x+ x**(3/2)
	if y > 1:
		y = y - trunc(y) 
	return y	

def f4(x):
	y = 2*x + (12*x)/(1-x)
	if y > 1:
		y = y - trunc(y) 
	return y