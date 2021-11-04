#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings; warnings.simplefilter('ignore')
import pandas as pd
df=pd.read_csv('interventionsEtInfos.csv')


# In[2]:


dflt = df.loc[df['annee']<2017]
dfv = df.loc[df['annee']==2017]


# In[3]:


y_val = dfv['nbInterventions'].copy()
y_val_pred = [dflt['nbInterventions'].mean()]*len(y_val)


# In[4]:


#get_ipython().run_line_magic('pylab', 'inline')
#pylab.rcParams['figure.figsize'] = (18,8)
from pylab import *
a,b = 60,90
#plot(range(a,b), y_val_pred[a:b], label="predicted")
plot(range(a,b), y_val[a:b], label="real")
plt.show()
legend()


# In[5]:

'''
from sklearn.metrics import mean_squared_error, mean_absolute_error

val_error = mean_squared_error(y_val, y_val_pred)
print("Validation RMSE:", sqrt(val_error))
print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))


# In[6]:


#get_ipython().run_line_magic('pylab', 'inline')
#pylab.rcParams['figure.figsize'] = (18,8)

g=df.loc[df['annee']<2017].groupby('heure')
plt.plot(range(0,24),[g.get_group(k)['nbInterventions'].sum() for k in range(0,24)])
plt.show()

# In[7]:


liste = [g.get_group(k)['nbInterventions'].mean() for k in range(0,24)]
y_moyenne = []
for k in range(len(y_val)):
    y_moyenne.append(liste[k%len(liste)])


# In[8]:


#get_ipython().run_line_magic('pylab', 'inline')
#pylab.rcParams['figure.figsize'] = (18,8)

a,b = 100,250
plot(range(a,b), y_moyenne[a:b], label="predicted")
plot(range(a,b), y_val[a:b], label="real")
legend()


# In[9]:


val_error = mean_squared_error(y_val, y_moyenne)
print("Validation RMSE:", sqrt(val_error))
print("Validation MAE:", mean_absolute_error(y_val, y_moyenne))


# In[ ]:


'''

