###
# Authors: Jered Dominguez-Trujillo and Soheila Jafari
# Description: Python Script to Aid in the Analysis of MLFlow Data for GEV_Assessment
# Currently Designed to Work with the MLFlow File Format and Data Located at /projects/bridges2016099/gev_assessment
# Date: May 26, 2020

# Import statements
import gevfit
import mlflow
import os
import pickle

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from statistics import mean
from scipy.stats import genextreme
from sklearn.utils import resample
from sklearn.utils import shuffle
from scipy import stats
from math import sqrt

# Returns a Dataframe of All Experiments Located in the MLRuns Directory
def getAllExperiments():
    df = pd.DataFrame(columns=['Experiment', 'expid', 'workload', 'distribution', 'a', 'b', 'cores', 'processors', 'stencil_size', 'osu', 'rabbit', 'rabbit_workload', 'iterations', 'machine', 'type', 'mode', 'container', 'verbose'])

    pth = "./mlruns/"
    dir_list = next(os.walk(pth))[1]
    dir_list = [int(i) for i in dir_list if '.trash' not in i]
    dir_list.sort()

    for eid in dir_list:
        df = df.append(parseExperiment(eid), ignore_index=True)

    return df


# Returns a Dataframe of All Runs for a Particular Experiment ID
def parseExperiment(eid):
    df = pd.DataFrame(columns=['Experiment', 'expid', 'workload', 'distribution', 'a', 'b', 'cores', 'processors', 'stencil_size', 'osu', 'rabbit', 'rabbit_workload', 'iterations', 'machine', 'type', 'mode', 'container', 'verbose'])
    pth = "./mlruns/" + str(eid)

    expdict = {}
    for root, dirs, files in os.walk(pth):
        path = root.split(os.sep)
        for file in files:
            if len(path) > 3:
                expid = path[3]

                if expid in expdict:
                    explist = expdict[expid]
                else:
                    explist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    explist[1] = expid

                if 'params' in root:

                    expid = path[3]

                    explist[0] = eid
                    explist[1] = expid
                    if file == 'workload':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        workload = f.read().strip()
                        explist[2] = workload
                    if file == 'a':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        a = int(f.read().strip())
                        explist[4] = a
                    if file == 'b':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        b = int(f.read().strip())
                        explist[5] = b
                    if file == 'cores':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        ncores = int(f.read().strip())
                        explist[6] = ncores
                    if file == 'processes':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        nprocesses = int(f.read().strip())
                        explist[7] = nprocesses
                    if file == 'stencil_size':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        stencil_size = int(f.read().strip())
                        explist[8] = stencil_size
                    if file == 'osu':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        osu = int(f.read().strip())
                        explist[9] = osu
                    if file == 'rabbit':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        rabbit = int(f.read().strip())
                        explist[10] = rabbit
                    if file == 'rabbit_workload':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        rabbit_workload = int(f.read().strip())
                        explist[11] = rabbit_workload
                    if file == 'iterations':
                        f = open(os.path.join(*path)+ '/' + file, "r")
                        iterations = int(f.read().strip())
                        explist[12] = iterations

                    # Experiments 1-319 - Sleep and fwq workloads were Gaussian distribution
                    if eid in range(1, 320):
                        if explist[2] == 'sleep' or explist[2] == 'fwq':
                            explist[3] = 'Gaussian'
                        else:
                            explist[3] = 'None'

                    # Experiments 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 50-84 ran on Wheeler
                    if eid in [1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]:
                        explist[13] = 'Wheeler'
                        explist[14] = 'None'
                        explist[15] = 'None'
                    # Experiments 3, 8, 9, 14, 15, 16, 17, 18, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49  ran on Stampede
                    elif eid in [3, 8, 9, 14, 15, 16, 17, 18, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
                        explist[13] = 'Stampede'
                        if eid in [9, 18, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
                            explist[14] = 'Skylake'
                            explist[15] = 'None'
                        else:
                            explist[14] = 'KNL'
                            if eid in [17]:
                                explist[15] = 'Flat'
                            else:
                                explist[15] = 'Cache'
                    # Experiments 235 - 319 ran on Attaway
                    elif eid in range(235, 320):
                        explist[13] = 'Attaway'
                        explist[14] = 'None'
                        explist[15] = 'None'
                    # Rest are ran on Cori
                    else:
                        explist[13] = 'Cori'
                        explist[14] = 'None'
                        explist[15] = 'None'

                    # Experiments 1-7 are ran in the container
                    if 0 < eid < 8:
                        explist[16] = 1
                    else:
                        explist[16] = 0

                    # Experiments 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 19-319  are verbose
                    if eid in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15] + list(range(19, 320))):
                        explist[17] = 1
                    else:
                        explist[17] = 0

                    expdict[expid] = explist

    for key in expdict:
        df.loc[len(df)] = expdict[key]

    return df


# Given a Row from parseExperiment or getAllExperiments or a Path, Return the Artifact Data as a DataFrame
def getData(row):
    if isinstance(row, pd.DataFrame)  or isinstance(row, pd.Series):
        eid = row['Experiment'].values
        rid = row['expid'].values
        rid = rid[0].replace("'", "")
        pth = './mlruns/' + str(eid[0]) + '/' + str(rid) + '/artifacts/bsp-trace.json'

        data = pd.read_json(pth, orient='columns')
    elif isinstance(row, str):
        data = pd.read_json(row, orient='columns')
    else:
        raise ValueError('Input to getData must be a DataFrame, Series, or String')

    return data


# Three Projection Methods

# Block Maxima Method
def block_maxima_project(samples, iterations, k, col='workload_max_usec'):
    if isinstance(samples, pd.Series):
        samples = pd.DataFrame([samples])
    if isinstance(samples, pd.DataFrame):
        if 'Experiment' in samples.columns:
            samples = getData(samples)
            samples = samples[samples['rank'] == 0]
            samples = samples[col]
        else:
            samples = samples[samples['rank'] == 0]
            samples = samples[col]

    count = 0
    new = [None] * int(np.ceil(iterations / k))

    for i in range(0, iterations, k):
        new[count] = max(samples[i:i + k])
        count += 1

    new = [x for x in new if x is not None]

    return new


# Resampling a GEV Distribution Method
def gev_resample_project(samples, nsamples, k):
    shape, loc, scale = gevfit.fit(samples)
    return [max(genextreme.rvs(shape, loc, scale, k)) for x in range(nsamples)]


# Resampling Actual Data Method
def resample_project(samples, nsamples, k, col='workload_max_usec'):
    if isinstance(samples, pd.Series):
        samples = pd.DataFrame([samples])
    if isinstance(samples, pd.DataFrame):
        if 'Experiment' in samples.columns:
            samples = getData(samples)
            samples = samples[samples['rank'] == 0]
            samples = samples[col]
        else:
            samples = samples[samples['rank'] == 0]
            samples = samples[col]

    nsamples = len(samples)
    return [max(resample(samples, n_samples=k, replace=False)) for _ in range(nsamples)]


# Harmonic Expected Value
def _harmonic(a, b):
    if b-a == 1:
        return 1, int(a)
    m = (a+b)//2
    p, q = _harmonic(a,m)
    r, s = _harmonic(m,b)
    return p*s+q*r, q*s


def harmonic(n):
    return mpq(*_harmonic(1,n+1))


# EMMA Expected Maximum Value Method
def emma(dist, p):
	return dist.ppf((0.570376002)**(1/p))

# Half-Efficiency Calculation
def ehalf(runtime0, ranks, dist, iterations, eff=0.5):
    efficiency = lambda n: runtime0 / (emma(dist, n) * iterations)
    return sp.optimize.brentq(lambda n: efficiency(n) - eff, ranks, 500000000) * ranks



# An Exmaple
def main():
    # Get DataFrame of All Experiments
    df = getAllExperiments()

    # Get Only Runs from Experiment 15
    df_filtered = df[df['Experiment'] == 15]

    # Get Specific Run - No Rabbit Workload and No Stencil and 8 ppn on 2 nodes
    df_filtered = df_filtered[df_filtered['cores'] == 8]
    df_filtered = df_filtered[df_filtered['processors'] == 16] # 2 Nodes
    df_filtered = df_filtered[df_filtered['rabbit_workload'] == 0]
    df_filtered = df_filtered[df_filtered['stencil_size'] == 0]
    print(df_filtered)

    # Get Data from Specific Run
    data = getData(df_filtered)
    print(data.head())

    # Get Data Only from Rank 0
    data_rank0 = data[data['rank'] == 0]

    # Find Shape, Location, and Scale of Max Data
    shape, loc, scale = gevfit.fit(data_rank0['workload_max_usec'])
    print("Shape: ", shape, "\tLocation: ", loc, "\tScale: ", scale)

    # Get Overall Runtime
    runtime0 = data_rank0['interval_max_usec'].sum()
    dist = stats.genextreme(shape, loc, scale)
    print("Runtime: ", runtime0, "Microseconds at Initial ", data['comm_size'].iloc[0], " Ranks")

    # Projected Runtime at k = 8 -> 128 ranks (16 nodes)
    projected = emma(dist, 8) * data['iterations'].iloc[0]
    print("K = 8: Projected Runtime: ", projected, "\tProjected Efficiency: ", runtime0 / projected)

    # Get Projected Scale of When Efficiency Reaches 50%
    eh = ehalf(runtime0, data['comm_size'].iloc[0], dist, data['iterations'].iloc[0])
    print("Expect 50% Efficiency at: ", eh, " Ranks")


#main()
