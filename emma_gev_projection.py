import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


import analysis
import pickle
import pandas as pd

import os.path
from os import path
import os, glob

import numpy as np

import scipy
from scipy import stats
from scipy import special
from scipy.optimize import minimize
from scipy.stats import genextreme
from scipy.stats import norm
from math import factorial as fac
from fractions import Fraction as mpq
from sklearn.utils import resample
from scipy.stats import gaussian_kde




np.random.seed(123)

np.seterr(divide='ignore', invalid='ignore')

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




def plot_gev(data,nbins,shape,loc,scale,lmin,lmax,color,label,ax):
    print("fit: shape {}, location{}, scale {}".format(shape,loc,scale))
    xx = np.linspace(lmin, lmax, num=nbins*100)
    #yy = gev.pdf(xx, shape, loc, scale)
    yy=stats.genextreme.pdf(xx, shape, loc=loc, scale=scale)
    ax.plot(xx, yy,label=label,color=color)



def plot_mean_and_CI(lmean, llb, lub, lx, label=None, color_mean=None, color_shading=None):
    mean=np.array(lmean) / 1000000
    lb=np.array(llb) / 1000000
    ub=np.array(lub) / 1000000
    x=np.array(lx)


    plt.fill_between(x, ub, lb,color=color_shading, alpha=0.3)
    # plot the mean on top
    plt.plot(x, mean, color_mean,label=label)



def plot_mean_and_CI2(fig, lmean, llb, lub, lx, label=None, color_mean=None, color_shading=None):
    mean=np.array(lmean) / 1000000
    lb=np.array(llb) / 1000000
    ub=np.array(lub) / 1000000
    x=np.array(lx)
    # plot the shaded range of the confidence intervals
    gs = fig.add_gridspec(1, 8)
    ax1 = fig.add_subplot(gs[0, 0:6])
    ax2 = fig.add_subplot(gs[0, 7])

    ax1.fill_between(x, ub, lb,color=color_shading, alpha=0.3)
    # plot the mean on top
    ax1.plot(x, mean, color_mean,label=label)

    x_d = np.linspace(llb[-1], lub[-1], 100)
    density = sum(stats.norm(xi).pdf(x_d) for xi in dengta)

    ax2.fill_between(x_d, density / 1000000, alpha=0.5)
    ax2.axis([llb[-1],llb[-1],0,np.max(density)]);
    #ax2.xlim([llb[-1],llb[-1]])
    #ax2.ylim(0,np.max(density))
    #ax2.plot(dengta, np.full_like(dengta, -0.1), '|k', markeredgewidth=1)

    return ax1



def find_mean_and_CI(nodes, wtimes, p):
    # nodes and wtimes are numpy arrays
    medians=[]
    lb=[]
    ub=[]
    xx=[]
    unique_nodes = np.unique(nodes)
    last=0
    for x in unique_nodes:
        grouped_times = wtimes[np.where(nodes == x)]
        medians.append(np.median(grouped_times))
        ub.append(np.percentile(grouped_times, p))
        lb.append(np.percentile(grouped_times, 100-p))

        #ub.append(np.max(grouped_times))
        #lb.append(np.min(grouped_times))
        xx.append(x)
        last=x

    gb=grouped_times[np.where(grouped_times>=lb[-1])]
    ga=gb[np.where(gb<=ub[-1])]

    return medians,lb,ub,xx,ga



def test_projection(sample,_NNODES,_NRANKS,_NITER,_PROJ_NNODES,p):

    sone=np.reshape(sample, sample.shape[0]*sample.shape[1])

    # parameteric bootstrapping with mom
    mblock=[]
    lblock=[]
    pblock=[]
    emv_block=[]
    pwm_block=[]
    x=[]


    i=_NNODES
    # perform a series of intermediate projections
    while i <=_PROJ_NNODES+1:

        for j in range(50): # use a boostrap of size 30 per projection
            #stemp=np.random.permutation(sone)
            stemp=np.random.choice(sone, sone.shape[0], replace=True)
            sblock=np.reshape(stemp, (-1,_NITER))
            mx1=np.amax(sblock, axis=0)
            lblock.append(np.sum(mx1))

            pwm_block.append(em_pwm(mx1, _NNODES*_NRANKS, i*_NRANKS)*_NITER)
            pwmfit=pwm_fit(mx1)
            r = genextreme.rvs(pwmfit[0], loc=pwmfit[1], scale=pwmfit[2], size=i*_NITER)
            #reshape and take the max per iteration
            pwmblock=np.reshape(r, (-1,_NITER))
            mx2=np.amax(pwmblock, axis=0)
            # append the sum of maximumns
            pblock.append(np.sum(mx2))

            emv_block.append(emv(mx1, _NNODES*_NRANKS, i*_NRANKS)*_NITER)
            momfit=mom_fit(mx1)
            r = genextreme.rvs(momfit[0], loc=momfit[1], scale=momfit[2], size=i*_NITER)
            #reshape and take the max per iteration
            momblock=np.reshape(r, (-1,_NITER))
            mx2=np.amax(momblock, axis=0)
            # append the sum of maximumns
            mblock.append(np.sum(mx2))
            x.append(i*_NRANKS)
        i*=2


    temp_block = mblock.copy()
    temp_block.extend(pblock)
    temp_x = x.copy()
    temp_x.extend(x)

    # get medians and CI for both mom and pwm

    return temp_x, temp_block






def apply_correction(medians,lb,ub,xx,sample_workload, nnodes):

    idxnodes = np.where(np.array(xx) >= nnodes)
    idxnode=idxnodes[0][0]
    if xx[idxnode]-nnodes > 0:
        dist1=xx[idxnode]-xx[idxnode-1]
        pd2=(xx[idxnode]-nnodes)/dist1
        correction=madians[idxnode-1]+((medians[idxnode]-medians[idxnode-1])*pd2)
    else:
        correction=medians[idxnode]

    correction=sample_workload-correction
    medians=medians+correction
    lb=lb+correction
    ub=ub+correction

    return medians,lb,ub,xx,correction



def get_formated_data(df_platform_NoStencil, workload, processors,fname):

    if path.isfile(fname):
        data = np.load(fname)

    else:

        ppn= 32

        data_platform = analysis.getData(df_platform_NoStencil)
        data_platform = data_platform[['workload', 'node', 'rank', 'iteration', 'iterations', 'workload_usec']]
        data_platform_Iteration0 = data_platform[(data_platform['iteration'] == 0)]

        print(data_platform.head())

        nodes =  data_platform['node'].unique()
        node_counter = np.zeros((len(nodes), 1))

        iterations = data_platform['iterations'].iloc[0]
        data = np.zeros((int(processors / ppn), ppn, iterations))

        for item in range(0, len(data_platform_Iteration0)):
            cur = data_platform_Iteration0.iloc[item]

            for node in nodes:
                if node == cur['node']:
                    i = int(np.where(nodes == node)[0][0])
                    j = int(node_counter[i])
            node_counter[i] = node_counter[i] + 1

            curData = data_platform[data_platform['rank'] == cur['rank']]

            for k in range(0, iterations):
                temp = curData.iloc[k]
                data[i, j, k] = temp['workload_usec']
        # save data into a npy
        np.save(fname, data)

    return data




workloads=['dgemm']
#workloads=['sleep','fwq','dgemm','spmv','lammps','hpcg']
_FACTOR = 1000000
_ALPHA=0.05
p = ((1.0-(_ALPHA/2.0)) * 100  )

workload=workloads[0]
platform='Cori'


df_All = analysis.getAllExperiments()


if platform=='Cori':
        coriRange = list( range( 85, 235 ) ) + list( range( 330, 450 ) )
        df_platform = df_All.loc[ df_All[ 'Experiment' ].isin( coriRange ) ]
        df_runtimes = pd.read_csv('Results/CoriData.csv',usecols=['Workload', 'Ranks', 'Stencil', 'Runtime'])
elif platform=='Attaway':
        attawayRange = list( range( 235, 330 ) ) + list( range( 450, 550 ) )
        df_platform = df_All.loc[ df_All[ 'Experiment' ].isin( attawayRange ) ]
        df_runtimes = pd.read_csv('Results/AttawayData.csv',usecols=['Workload', 'Ranks', 'Stencil', 'Runtime'])
else:
        print('Error, unknown platform')
        exit(-1)


fn=1
for workload in workloads:

    # clean lists and constants
    s_medians=[]
    s_lb=[]
    s_ub=[]
    s_dengt=[]
    all_nodes=[]
    all_projections=[]
    all_densities=[]
    arr_den=np.empty(0)
    sample_workloads=[]
    total_workloads=[]
    _PROJ_NNODES=128

    # all this part runs with the 256 workload
    contlabel=0

    df_platform_new = df_platform[(df_platform['processors']==256) & (df_platform['workload']==workload) & (df_platform['stencil_size'] != 0) & (df_platform['rabbit_workload'] == 0)]

    for num in range(len(df_platform_new)):
        fname = 'Emma_Stencil/'+platform+'_'+workload+'_256_'+str(num)+'.npy'
        print(fname)
        sample_data = get_formated_data(df_platform_new.iloc[num], workload, 256, fname)

        _NITER=sample_data.shape[2]
        _NNODES = 1
        _SAMPLE_NNODES=sample_data.shape[0]
        _NRANKS=sample_data.shape[1]
        # compute stats for the workload
        sampleall_max=np.amax(np.amax(sample_data, axis=1),axis=0)
        sampleall_workload=np.sum(sampleall_max)
        sample_workloads.append(sampleall_workload)


        for i in range(_SAMPLE_NNODES):
            # select a sample of the workload of size _NNODES
            s1=sample_data[i,0:_NRANKS,0:_NITER]
            nodes, projections = test_projection(s1,_NNODES,_NRANKS,_NITER,_PROJ_NNODES,p)
            all_nodes.extend(nodes)
            all_projections.extend(projections)


        mediansa,lba,uba,xxa,dengta=find_mean_and_CI(np.array(all_nodes), np.array(all_projections), p)
        medians,lb,ub,xx,correction = apply_correction(mediansa, lba, uba, xxa, sampleall_workload, _SAMPLE_NNODES*_NRANKS)
        all_densities.extend(dengta+correction) # apply correction to density estimation

        # plot figures
        fig = plt.figure(fn, figsize=(7, 3))

        plot_mean_and_CI(medians, ub, lb, xx, color_mean='k', color_shading='b')
        if contlabel==0:
            plt.plot(_SAMPLE_NNODES*_NRANKS, sampleall_workload / 1000000, 'ko', label='Sample workload')
            contlabel+=1
        else:
            plt.plot(_SAMPLE_NNODES*_NRANKS, sampleall_workload / 1000000, 'ko')




    # all this part runs with > 256 experiments, they are used only for assessment

    df_rnew = df_runtimes[(df_runtimes['Workload']==workload) & (df_runtimes['Ranks']>256) & (df_runtimes['Stencil']!=0)]

    for num in range(len(df_rnew)):
        runtime = df_rnew.iloc[num]['Runtime']
        ranks = df_rnew.iloc[num]['Ranks']
        if num==0:
            plt.plot(ranks, runtime, 'ro', label='Scaled-up workload')
        else:
            plt.plot(ranks, runtime, 'ro')


    # plot the rest of the figure
    arrden=np.array(all_densities)
    x_d = np.linspace(np.min(arrden), np.max(arrden), 500)
    density = sum(norm(xi).pdf(x_d) for xi in arrden)
    density=((200)*density)/np.max(density)
    #plt.plot(density+(_PROJ_NNODES*_NRANKS+2), x_d / 1000000, label='Probability density estimation')

    if (platform=='Cori'):
        if (workload=='dgemm'):
            plt.ylim(300, 500)
        if (workload=='spmv'):
            plt.ylim(700, 850)
    if (platform=='Attaway'):
        if (workload=='dgemm'):
            plt.ylim(300, 500)
        if (workload=='spmv'):
            plt.ylim(520, 600)
        if (workload=='hpcg'):
            plt.ylim(700, 1400)
        if (workload=='lammps'):
            plt.ylim(1000, 2400)

    if (workload == 'sleep'):
        w = 'ftq'
    else:
        w = workload

    plt.xlabel('Number of Ranks')
    plt.ylabel('Runtime (s)')
    plt.legend(loc='lower right', borderaxespad=0.5)
    plt.title(platform+' '+w+' -  per node bootstrap, global CIs with Stencil')
    plt.tight_layout()
    #plt.show()
    fig.savefig('Emma_Stencil/figs/'+platform+'_'+workload+'.png')
    plt.close(fig)
    fn=fn+1


