import numpy as np
np.random.seed(123)
import gevfit

import pandas as pd
from scipy import stats
from scipy import special
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import scipy
from scipy.stats import genextreme
import os, glob
from math import factorial as fac
from fractions import Fraction as mpq
from sklearn.utils import resample

import seaborn as sns

# https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
def mle_fit(samples):
    def cost(params):
        shape, loc, scale = params[0], params[1], params[2]

        negLL = -np.sum(stats.genextreme.logpdf(samples,shape,loc,scale))

        return(negLL)

    guess = np.array([0,np.mean(samples),np.std(samples)])
    results = minimize(cost, guess, method='Nelder-Mead',
                      options={'disp': False})
    return(results['x'])


'''
    Gives the nth moment of the given samples
    xs : the samples
    moment: the moment wanted
'''
def moment(xs, moment):
    samples = np.sort(xs)
    n = samples.size
    res = 0
    for j, sample in enumerate(samples):
        term = sample
        for i in range(1,moment + 1):
            term = term * (j + 1 - i) / (n - i)
        res = res + term
    return res / n

'''
    Uses Hosking, et. al's PWM method for estimating GEV parameters
'''
def pwm_fit(samples):
    # Generate moments from sample data
    samples = np.sort(samples)
    n = samples.size
    b0 = moment(samples,0)
    b1 = moment(samples,1)
    b2 = moment(samples,2)

    # Generate GEV parameters
    c = (2*b1-b0)/(3*b2-b0) - np.log(2)/np.log(3)
    est_shape = 7.8590*c+2.9554*np.power(c,2)
    gamma = special.gamma(1 + est_shape)
    est_scale = est_shape*(2*b1-b0)/(gamma*(1-np.power(2,-est_shape)))
    est_loc = b0 + est_scale*(gamma-1)/est_shape
    return est_shape,est_loc,est_scale



def mom_fit(samples):
    mean = np.mean(samples)
    std = np.std(samples)
    skew = stats.skew(samples)
    
    def cost(shape):
        numerator = -special.gamma(1+3*shape)+ \
                    3*special.gamma(1+shape)*special.gamma(1+2*shape)- \
                    2*special.gamma(1+shape)**3
        denominator = (special.gamma(1+2*shape) - \
                       special.gamma(1+shape)**2)**1.5
        cost = (np.sign(shape)*numerator/denominator - skew)**2
        return(cost)
    guess = 0
    shape = minimize(cost, guess, method='Nelder-Mead',
                          options={'disp': False})['x'][0]
    scale = std*np.abs(shape)/ \
            (special.gamma(1+2*shape) - \
             special.gamma(1+shape)**2)**0.5
    loc = mean - scale*(1-special.gamma(1+shape))/shape
    return(shape,loc,scale)


def emv(samples, num_samples_base, num_samples_big):
    shape,loc,scale = mom_fit(samples)
    phi = 0.570376002
    n = num_samples_big / num_samples_base
    emma = stats.genextreme.ppf(phi**(1/(n)),shape,loc,scale)
    return emma


def em_pwm(samples, num_samples_base, num_samples_big):
    shape,loc,scale = pwm_fit(samples)
    phi = 0.570376002
    n = num_samples_big / num_samples_base
    emma = stats.genextreme.ppf(phi**(1/(n)),shape,loc,scale)
    return emma


def gev_project(params, k, samples=1000):
    shape, loc, scale = params
    project_samples = [ max(genextreme.rvs(shape, loc, scale, k)) for x in range(k*samples)]
    return gevfit.fit(project_samples)


def gev_project2(maxes, k):
    project_samples = [ max(resample(maxes, n_samples=k, replace=False)) for _ in range(k*len(maxes))]
    return gevfit.fit(project_samples)



def plot_data(data,nbins):
    # matplotlib histogram
    plt.hist(data, color = 'blue', edgecolor = 'black',
             bins = nbins)

    # seaborn histogram
    sns.distplot(data, hist=True, kde=False, 
                 bins=nbins, color = 'blue',
                 hist_kws={'edgecolor':'black'})

    
def plot_density(data, nbins):    
    # Density Plot and Histogram of all arrival delays
    sns.distplot(data, hist=True, kde=True, 
             bins=nbins, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    
    
def plot_gev(data,nbins,shape,loc,scale,lmin,lmax):

    print("fit: shape {}, location{}, scale {}".format(shape,loc,scale))
    xx = np.linspace(lmin, lmax, num=nbins*10)
    #yy = gev.pdf(xx, shape, loc, scale)
    yy=stats.genextreme.pdf(xx, shape, loc=loc, scale=scale)
    
    plt.plot(xx, yy, color = 'darkred')
    plt.show()
    
    


_NNODES = 32
_NRANKS = 32
_NITER = 100
_PROJ_NNODES = 1024
_ALPHA=0.05

p = ((1.0-_ALPHA)/2.0) * 100  


#pop = np.random.randint(0,500 , size=1000)
#sample = np.random.choice(pop, size=300) #so n=300


# generate samples

a = 1. # shape
s = np.random.weibull(a, (_PROJ_NNODES,_NRANKS,_NITER))


#s = np.random.normal(loc=5.0, scale=1.0, size=(_PROJ_NNODES,_NRANKS,_NITER))
# add per node disturbances


# compute stats for all thr workload
sall=np.reshape(s, (-1,_NITER)) 
# get maxima
all_max=np.amax(sall, axis=0)
all_workload=np.sum(all_max)
print("all workload {}".format(all_workload))

all_reshape=np.reshape(all_max,(-1,1))
print(all_reshape.shape)
all_gev = gevfit.fit(all_reshape)
print("all gev {}".format(all_gev))
all_pwm = pwm_fit(all_reshape)
print("all pwm {}".format(all_pwm))
all_mom = mom_fit(all_reshape)
print("all mom {}".format(all_mom))


s1=s[0:_NNODES,0:_NRANKS,0:_NITER]
sone=np.reshape(s1, (-1,_NITER)) 
one_workload=np.sum(np.amax(sone, axis=0))
print("one workload no B {}".format(one_workload))
    
sone=np.reshape(s1, (-1,1)) 



# parameteric bootstrapping with mom
mblock=[]
lblock=[]
emv_block=[]


x=[]
for i in range(_NNODES,_PROJ_NNODES+1,_NNODES):
    print(i)

    for j in range(30):
        stemp=np.random.permutation(sone)
        sblock=np.reshape(stemp, (-1,_NITER)) 
        mx1=np.amax(sblock, axis=0)
        lblock.append(np.sum(mx1))
        
        emv_block.append(emv(mx1, _NNODES, i)*_NITER)
        momfit=mom_fit(mx1)
        r = genextreme.rvs(momfit[0], loc=momfit[1], scale=momfit[2], size=i*_NITER)
        #reshape and take the max per iteration
        momblock=np.reshape(r, (-1,_NITER)) 
        mx2=np.amax(momblock, axis=0) 
        # append the sum of maximumns
        mblock.append(np.sum(mx2))
        x.append(i*_NRANKS)


arr_block=np.array(lblock)

# get donfidence intervals using quantiles      
lowCI=np.percentile(arr_block, p)
#highCI=np.percentile(arr_block, 100-p)
block_workload=sum(lblock)/len(lblock)
print("one workload with B {}".format(block_workload))
#print("[{},{}]".format(lowCI,highCI))

    
arr_block=np.array(mblock)
#lowCI=np.percentile(arr_block, p)
highCI=np.percentile(arr_block, 100-p)
print("mom workload with B {}".format(sum(mblock)/len(mblock)))
print("[{},{}]".format(lowCI,highCI))

emv_workload=emv_block[-1]
print("emma workload {}".format(emv_workload))

print("all workload {}".format(all_workload))