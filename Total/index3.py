import pylab as pp
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import scipy.optimize as optim
import matplotlib.pyplot as plt

def fitfunc(t,r,p):
    'Function that returns Ca computed from an ODE for a k'
    def myode(Ca, t):
        return r*(Ca**p)

    Ca0 = y[0]
    Casol = odeint(myode, Ca0, t)
    return Casol[:,0]

data = pd.read_csv("data.csv",sep=',')
x = data['Date']
y = data['Total Cases']
x = np.array(x)
y = np.array(y)

params = [0.6,0.5]
bounds = ([0,0],[np.inf,1])

print('Data Loaded')

(r,p),cov = optim.curve_fit(fitfunc, x, y, p0=params,bounds = bounds)
print('First Estimate Done')

# plt.scatter(x,y,edgecolors='k',facecolors='none',label='Reported')
plt.plot(x,y,'ko',mfc="none",label='Reported',color='k')


# 200 Simulations for confidence intervals of Params And Y values(Total Confirmed)

x = np.append(x,range(72,82))
print('Days Extended')

# Using our estimates generate extended dataframe
y = fitfunc(x,r,p)

y_total = []# mat containing simulations for error bar
r_ar = []#For 95% ci param_ r
p_ar = []#For 95% ci param_ co
# 200 simulations and param estimation
print('Simulations Begin')
for trial in range(1,200):
    print('Simulation: ',trial)
    y_dash = [y[0]]
    for i in range(1, len(y)):
        tmp = np.random.poisson(y[i] - y[i-1])+y_dash[i-1]
        y_dash.append(tmp)
    
    (b,c),cov = optim.curve_fit(fitfunc, x, y_dash, p0=params,bounds = bounds)
    r_ar.append(b)
    p_ar.append(c)
    y_total.append(y_dash)

# This question isn't related to Python. I think you need to read an intro to bootstrapping.
#  "An Introduciton to Statistical Learning" provides a good one. 
#  The idea is not to sample 100 -- you must sample with replacement and taking the same sample size (500).
#   Yes, then you reestimate your parameter many times. And then there's several ways of taking all 
#   of these estimates and turning them into a confidence interval. For example, you can use them to
#    estimate the standard error (the standard deviation of the sampling distribution),
#     and then use +/- 2*se.

y_total = np.transpose(y_total)#transpose
yerr = []
for i in range(len(y_total)):
    yerr.append(np.std(y_total[i])*2)
print('Yerr calculated')

print('Writing Results to Res1.txt')
# Storing Results
with open('Res3_GGM.txt', 'w') as file:
    file.write('95% CI for Params\n')

    file.writelines(['\nParam r ',str(r),'\n'])
    file.writelines([str(r-(np.std(r_ar)*2)),'  ', str((np.std(r_ar)*2)+r),'\n'])

    file.writelines(['\nParam p ',str(p),'\n'])
    file.writelines([str(p-(np.std(p_ar)*2)),'  ',str((np.std(p_ar)*2)+p),'\n\n'])

    file.writelines(['Day','Total Cases', '95% Confidence Interval(+-)\n'])
    for i in range(len(y)):
        file.writelines([str(x[i]),'  ',str(y[i]),'  ',str(y[i]+yerr[i]),'  ',str(y[i]-yerr[i]),'\n'])

print('Plotting Graph')
# Scale x5 for showing relative errors in graph
yerr = [i*5 for i in yerr]    


# plt.plot(x,Logistic_Paper(x,K,r,co))
plt.errorbar(x,y,yerr=yerr,capsize=2,fmt='m--',label=f"Generalized Growth Model\np={str(p)[:4]}, r={round(r,4)}")
plt.legend(framealpha=1, frameon=True)
plt.xlabel('No of Days')
plt.ylabel('Total Cases')
plt.figtext(0.99, 0.01, 'Jan 31,2020 to Apr 20,2020', horizontalalignment='right')
plt.grid()
plt.show()



