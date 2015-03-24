# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/sara/.spyder2/.temp.py
"""

#import os

#os.chdir("/home/sara/intro-data-science/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm


df = pd.read_csv("../data/train.csv")
#print df.head()
#print df.dtypes
#print df['Age'].describe()


sex =df.groupby(['Sex', 'Survived']).size().unstack()
sex = sex.reindex(index=sex.index[::-1])
x_lab = ['Male', 'Female']
x_locs = [.4, 1.4]

p1 = plt.bar([0, 1], sex[0], color='m', label='Perished')
p2 = plt.bar([0, 1], sex[1], color='teal', bottom=sex[0], label='Survived')
plt.xticks(x_locs, x_lab, fontsize=20)
plt.tick_params(labelsize=20) 
plt.ylabel('Number of Passengers', fontsize=20)
plt.legend([p2[0], p1[0]], ['Survived', 'Perished'], fontsize=20)
#plt.show()

fig = plt.gcf()
fig.set_size_inches(10,8)
plt.savefig('../images/gender_plot.png')

df = df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
#df = df.dropna()
#find median age of 28.0
med_age = df.Age.median()
df.Age = df.Age.fillna(med_age)



# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")    

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(df.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel desnsity estimate of the subset of the 1st class passanges's age
df.Age[df.Pclass == 1].plot(kind='kde')    
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(df.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")

plt.savefig('../images/ex_plots.png')


fig = plt.figure(figsize=(18,6))

# create a plot of two subsets, male and female, of the survived variable.
# After we do that we call value_counts() so it can be easily plotted as a bar graph. 
# 'barh' is just a horizontal bar graph
ax1 = fig.add_subplot(121)
df.Survived[df.Sex == 'male'].value_counts().plot(kind='barh',label='Male')
df.Survived[df.Sex == 'female'].value_counts().plot(kind='barh', color='#ff00ff',label='Female')
ax1.set_ylim(-1, 2) 
plt.title("Who survived with respect to gender"); plt.legend(loc='best')





