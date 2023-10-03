#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:15:16 2021

For SAREQ; functions pulled from Gorner as needed

@author: theresasawi
"""



import h5py
import pandas as pd
import numpy as np
#from obspy import read
from matplotlib import pyplot as plt

import fiona

import geopandas as gpd


import datetime as dtt
import cartopy.feature as cfeature

import datetime
from scipy.stats import kurtosis
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from scipy import spatial
import scipy.stats as stats
from haversine import haversine
from scipy.signal import butter, lfilter
#import librosa
# # sys.path.insert(0, '../01_DataPrep')
from scipy.io import loadmat
from sklearn.decomposition import PCA
# sys.path.append('.')
from sklearn.metrics import silhouette_samples
import scipy as sp
import scipy.signal

from scipy import stats

from obspy.signal.cross_correlation import correlate, xcorr_max

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sklearn.metrics

from scipy.spatial.distance import euclidean

from matplotlib.ticker import FormatStrFormatter





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def removeEndedLateRES(cat, endYear = 2015):
    
    startYear = 1984
    
    cat_reloc_graded_stats_cont = pd.DataFrame()

    for cl in np.unique(cat.RID2):
        df_cl = cat[cat.RID2==cl].copy()

        time_mean = df_cl.time_mean_yr.iloc[0] *3

        time_end = df_cl.index[-1] 
        time_start = df_cl.index[0] 
        

        if ((time_end.date()+pd.Timedelta(time_mean*365.25,'day')).year >= endYear) and ((time_start.date()-pd.Timedelta(time_mean*365.25,'day')).year <= startYear):



            cat_reloc_graded_stats_cont = cat_reloc_graded_stats_cont.append(df_cl)

        
#     print(len(cat),len(cat_reloc_graded_stats_cont))
#     print(len(cat)-len(cat_reloc_graded_stats_cont), ' events removed')        
    
    return cat_reloc_graded_stats_cont  


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def removeEndedLateRES_RID(cat, endYear = 2015):
    
    startYear = 1984
    
    cat_reloc_graded_stats_cont = pd.DataFrame()

    for cl in np.unique(cat.RID):
        df_cl = cat[cat.RID==cl].copy()

        time_mean = df_cl.time_mean_yr.iloc[0] *3

        time_end = df_cl.index[-1] 
        time_start = df_cl.index[0] 
        

        if ((time_end.date()+pd.Timedelta(time_mean*365.25,'day')).year >= endYear) and ((time_start.date()-pd.Timedelta(time_mean*365.25,'day')).year <= startYear):



            cat_reloc_graded_stats_cont = cat_reloc_graded_stats_cont.append(df_cl)

        
    print(len(cat),len(cat_reloc_graded_stats_cont))
    print(len(cat)-len(cat_reloc_graded_stats_cont), ' events removed')        
    
    return cat_reloc_graded_stats_cont  

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotScatterMapRID_basic(cat_raw, scale = .01,plotNew = 1,ax=None):

    if ax is None:
        ax = plt.gca()


    radii = [scale*calcRadius(m)**2 for m in cat_raw.magnitude]
    cat_raw['radius'] = radii


    ## Split into magnitude bins - Background seismicity
    cat_0 = cat_raw[cat_raw.magnitude<1]

    cat_1 = cat_raw[cat_raw.magnitude>=1]
    cat_1 = cat_1[cat_1.magnitude<2]

    cat_2 = cat_raw[cat_raw.magnitude>=2]
    cat_2 = cat_2[cat_2.magnitude<3]

    cat_3 = cat_raw[cat_raw.magnitude>=3]



    #Background
    s0=1
    s1=10
    s2=100
    s3=500


    ax.scatter(cat_0.long, cat_0.lat,edgecolor='steelblue',facecolor='None',s=s0,label='Mw0')
    ax.scatter(cat_1.long, cat_1.lat,edgecolor='steelblue',facecolor='None',s=s1,label='Mw1')
    ax.scatter(cat_2.long, cat_2.lat,edgecolor='steelblue',facecolor='None',s=s2,label='Mw2')
    ax.scatter(cat_3.long, cat_3.lat,edgecolor='steelblue',facecolor='None',s=s3,label='Mw3')

    ax.grid('on')

    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotScatterMapRID(cat_raw, scale = .01,plotNew = 1,ax=None):

    if ax is None:
        ax = plt.gca()


    radii = [scale*calcRadius(m)**2 for m in cat_raw.magnitude]
    cat_raw['radius'] = radii

    cat = pd.DataFrame()

    for clus in cat_raw.drop_duplicates('RID').RID[1:]: #skip first for 9999
        df = cat_raw[cat_raw.RID==clus]
        medlat = np.median(df.lat)
        medlong = np.median(df.long)        
        medmag = np.median(df.magnitude)   
        cat = cat.append({'RID':clus,
                          'lat':medlat,
                           'long':medlong,
                           'magnitude':medmag},
                          ignore_index=True)
                
    
    cat = cat.append(cat_raw[cat_raw.RID=='9999'])
    
    cat_rep = cat[cat.RID!='0']
    cat_rep = cat_rep[cat_rep.RID!='9999']
    cat_new_raw = cat_raw[cat_raw.RID=='0']
    cat_BG = cat_raw[cat_raw.RID=='9999']
    
    cat_new = pd.DataFrame()
    print('cat_new_raw',len(cat_new_raw))
    print('cat_new_raw',len(cat_new_raw.drop_duplicates('RID2')))

    for clus in cat_new_raw.drop_duplicates('RID2').RID2: 
        df = cat_new_raw[cat_new_raw.RID2==clus]
        medlat = np.median(df.lat)
        medlong = np.median(df.long)        
        medmag = np.median(df.magnitude)   
        cat_new = cat_new.append({'RID2':clus,
                          'lat':medlat,
                           'long':medlong,
                           'magnitude':medmag},
                          ignore_index=True)

    print('cat_new',len(cat_new))

    ## Split into magnitude bins - Background seismicity
    cat_0 = cat_BG[cat_BG.magnitude<1]

    cat_1 = cat_BG[cat_BG.magnitude>=1]
    cat_1 = cat_1[cat_1.magnitude<2]

    cat_2 = cat_BG[cat_BG.magnitude>=2]
    cat_2 = cat_2[cat_2.magnitude<3]

    cat_3 = cat_BG[cat_BG.magnitude>=3]


    ## Split into magnitude bins - RES seismicity
    cat_rep_0 = cat_rep[cat_rep.magnitude<1]

    cat_rep_1 = cat_rep[cat_rep.magnitude>=1]
    cat_rep_1 = cat_rep_1[cat_rep_1.magnitude<2]

    cat_rep_2 = cat_rep[cat_rep.magnitude>=2]
    cat_rep_2 = cat_rep_2[cat_rep_2.magnitude<3]

    cat_rep_3 = cat_rep[cat_rep.magnitude>=3]
    
    
    ## Split into magnitude bins - NEW seismicity
    cat_new_0 = cat_new[cat_new.magnitude<1]

    cat_new_1 = cat_new[cat_new.magnitude>=1]
    cat_new_1 = cat_new_1[cat_new_1.magnitude<2]

    cat_new_2 = cat_new[cat_new.magnitude>=2]
    cat_new_2 = cat_new_2[cat_new_2.magnitude<3]

    cat_new_3 = cat_new[cat_new.magnitude>=3]    




    #Background
    s0=1
    s1=10
    s2=100
    s3=500


    ax.scatter(cat_0.long, cat_0.lat,edgecolor='grey',facecolor='None',s=s0,label='Mw0')
    ax.scatter(cat_1.long, cat_1.lat,edgecolor='grey',facecolor='None',s=s1,label='Mw1')
    ax.scatter(cat_2.long, cat_2.lat,edgecolor='grey',facecolor='None',s=s2,label='Mw2')
    ax.scatter(cat_3.long, cat_3.lat,edgecolor='grey',facecolor='None',s=s3,label='Mw3')
    
    ax.scatter(cat_rep_0.long, cat_rep_0.lat,edgecolor='b',facecolor='None',s=s0)
    ax.scatter(cat_rep_1.long, cat_rep_1.lat,edgecolor='b',facecolor='None',s=s1)
    ax.scatter(cat_rep_2.long, cat_rep_2.lat,edgecolor='b',facecolor='None',s=s2)
    ax.scatter(cat_rep_3.long, cat_rep_3.lat,edgecolor='b',facecolor='None',s=s3)
    
    if plotNew:
        ax.scatter(cat_new_0.long, cat_new_0.lat,edgecolor='r',facecolor='None',s=s0)
        ax.scatter(cat_new_1.long, cat_new_1.lat,edgecolor='r',facecolor='None',s=s1)
        ax.scatter(cat_new_2.long, cat_new_2.lat,edgecolor='r',facecolor='None',s=s2)
        ax.scatter(cat_new_3.long, cat_new_3.lat,edgecolor='r',facecolor='None',s=s3)  
    

    
  

    ax.grid('on')


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

    
def rotateMapView(cat,angle,lat0,lon0):

    lat = cat.lat
    lon = cat.long

    dlat = lat - lat0
    dlon = lon - lon0

    y= dlat*111.1949;
    x= dlon*np.cos(lat*np.pi/180)*111.1949;

    # dlat and dlon being the difference between lat/lon origin and earthquake epicenter. 
    # lat is the latitude of the earthquake location (since longitude is latitude dependent). 

    #Then rotate, with PHI being the angle you like to rotate about (clockwise from north):
    phiA= (angle-90)*np.pi/180;

    # then plot distance along fault vs. depth: 
    xx= x * np.cos(phiA)- y * np.sin(phiA)

#     cat['dist_along_strike_km'] = xx
    # plt.scatter(xx,cat.depth_km)
    
    return xx

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotScatterMapDepthAlongStrikeRID(cat_raw,lat0,lon0,angle = 48, scale = .01,plotNew=1,ax=None):

    if ax is None:
        ax = plt.gca()

    cat_raw['dist_along_strike_km'] = rotateMapView(cat_raw,angle,lat0,lon0)

        
    radii = [scale*calcRadius(m)**2 for m in cat_raw.magnitude]
    cat_raw['radius'] = radii
    
    cat = pd.DataFrame()

    for clus in cat_raw.drop_duplicates('RID').RID[1:]: #skip first for 9999
        df = cat_raw[cat_raw.RID==clus].copy()
        medDepth = np.median(df.depth_km)
        medDistAlongStrike = np.median(df.dist_along_strike_km)        
        medmag = np.median(df.magnitude)   
        cat = cat.append({'RID':clus,
                          'depth_km':medDepth,
                           'dist_along_strike_km':medDistAlongStrike,
                           'magnitude':medmag},
                          ignore_index=True)

    
    cat = cat.append(cat_raw[cat_raw.RID=='9999'])

    #Get already cataloged REQS
    cat_rep = cat[cat.RID!='0']
    cat_rep = cat_rep[cat_rep.RID!='9999']
    cat_new_raw = cat_raw[cat_raw.RID=='0']
    cat_BG = cat_raw[cat_raw.RID=='9999']
    
    cat_new = pd.DataFrame()


    for clus in cat_new_raw.drop_duplicates('RID2').RID2: 
        df = cat_new_raw[cat_new_raw.RID2==clus].copy()
        medDepth = np.median(df.depth_km)
        medDistAlongStrike = np.median(df.dist_along_strike_km)       
        medmag = np.median(df.magnitude)   
        cat_new = cat_new.append({'RID2':clus,
                          'depth_km':medDepth,
                           'dist_along_strike_km':medDistAlongStrike,
                           'magnitude':medmag},
                          ignore_index=True)


    ## Split into magnitude bins - Background seismicity
    cat_0 = cat_BG[cat_BG.magnitude<1]

    cat_1 = cat_BG[cat_BG.magnitude>=1]
    cat_1 = cat_1[cat_1.magnitude<2]

    cat_2 = cat_BG[cat_BG.magnitude>=2]
    cat_2 = cat_2[cat_2.magnitude<3]

    cat_3 = cat_BG[cat_BG.magnitude>=3]


    ## Split into magnitude bins - RES seismicity
    cat_rep_0 = cat_rep[cat_rep.magnitude<1]

    cat_rep_1 = cat_rep[cat_rep.magnitude>=1]
    cat_rep_1 = cat_rep_1[cat_rep_1.magnitude<2]

    cat_rep_2 = cat_rep[cat_rep.magnitude>=2]
    cat_rep_2 = cat_rep_2[cat_rep_2.magnitude<3]

    cat_rep_3 = cat_rep[cat_rep.magnitude>=3]



    ## Split into magnitude bins - RES seismicity
    cat_new_0 = cat_new[cat_new.magnitude<1]

    cat_new_1 = cat_new[cat_new.magnitude>=1]
    cat_new_1 = cat_new_1[cat_new_1.magnitude<2]

    cat_new_2 = cat_new[cat_new.magnitude>=2]
    cat_new_2 = cat_new_2[cat_new_2.magnitude<3]

    cat_new_3 = cat_new[cat_new.magnitude>=3]
    #Background
    s0=1
    s1=10
    s2=100
    s3=500


    ax.scatter(cat_0.dist_along_strike_km, cat_0.depth_km,edgecolor='gray',facecolor='None',s=s0,label='Mw0')
    ax.scatter(cat_1.dist_along_strike_km, cat_1.depth_km,edgecolor='gray',facecolor='None',s=s1,label='Mw1')
    ax.scatter(cat_2.dist_along_strike_km, cat_2.depth_km,edgecolor='gray',facecolor='None',s=s2,label='Mw2')
    ax.scatter(cat_3.dist_along_strike_km, cat_3.depth_km,edgecolor='gray',facecolor='None',s=s3,label='Mw3')
    
    ax.scatter(cat_rep_0.dist_along_strike_km, cat_rep_0.depth_km,edgecolor='b',facecolor='None',s=s0)
    ax.scatter(cat_rep_1.dist_along_strike_km, cat_rep_1.depth_km,edgecolor='b',facecolor='None',s=s1)
    ax.scatter(cat_rep_2.dist_along_strike_km, cat_rep_2.depth_km,edgecolor='b',facecolor='None',s=s2)
    ax.scatter(cat_rep_3.dist_along_strike_km, cat_rep_3.depth_km,edgecolor='b',facecolor='None',s=s3)
    
    if plotNew:
        ax.scatter(cat_new_0.dist_along_strike_km, cat_new_0.depth_km,edgecolor='r',facecolor='None',s=s0)
        ax.scatter(cat_new_1.dist_along_strike_km, cat_new_1.depth_km,edgecolor='r',facecolor='None',s=s1)
        ax.scatter(cat_new_2.dist_along_strike_km, cat_new_2.depth_km,edgecolor='r',facecolor='None',s=s2)
        ax.scatter(cat_new_3.dist_along_strike_km, cat_new_3.depth_km,edgecolor='r',facecolor='None',s=s3)   
    

 

    ax.grid('on')
    ax.invert_yaxis()
    
    return cat


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotCumul5(cat_clust,size=50,ax=None):
    
    if ax is None:
        ax = plt.gca()
        
    
    cat_clust = cat_clust.sort_index()   
    cumsum = [i for i in range(1,len(cat_clust)+1)]
    dates = [pd.to_datetime(d) for d in cat_clust.index]    
    
    cat_clust_old = cat_clust[cat_clust.RID!='0'].sort_index()   
    cumsum_old = [i for i in range(1,len(cat_clust_old)+1)]
    dates_old = [pd.to_datetime(d) for d in cat_clust_old.index]

    
    cat_clust_new = cat_clust[cat_clust.RID=='0'].sort_index()   
    cumsum_new = [i for i in range(1,len(cat_clust_new)+1)]
    dates_new = [pd.to_datetime(d) for d in cat_clust_new.index]

    ax.step(dates, cumsum,color='k',where='post')
    ax.scatter(dates, cumsum,c='k',s=size)
        
    ax.set_ylabel('Cumalitive number of earthquakes')    

    ax.tick_params(axis='x', labelrotation = 90,size=10) 
    
    ax.set_ylim(ymin=0)
    
    ax.grid('on')
        
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def linearizeFP(SpecUFEx_H5_path,ev_IDs):
    """
    Linearize fingerprints, stack into array 

    Parameters
    ----------
    SpecUFEx_H5_path :str
    cat00 : pandas dataframe

    Returns
    -------
    X : numpy array
        (Nevents,Ndim)

    """

    X = []
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        for evID in ev_IDs:
            fp = MLout['fingerprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)

    X = np.array(X)

    return X




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def PVEofPCA(X,numPCMax=100,cum_pve_thresh=.86,stand='MinMax',verbose=1):
    """
    Calculate cumulative percent variance explained for each principal component.

    Parameters
    ----------
    X : numpy array
        Linearized fingerprints.
    numPCMax : int, optional
        Maximum number of principal components. The default is 100.
    cum_pve_thresh : int or float, optional
        Keep PCs until cumulative PVE reaches threshold. The default is .8.
    stand : str, optional
        Parameter for SKLearn's StandardScalar(). The default is 'MinMax'.

    Returns
    -------
    PCA_df : pandas dataframe
        Columns are PCs, rows are event.
    numPCA : int
        Number of PCs calculated.
    cum_pve : float
        Cumulative PVE.

    """


    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    numPCA_range = range(1,numPCMax)


    for numPCA in numPCA_range:

        sklearn_pca = PCA(n_components=numPCA)

        Y_pca = sklearn_pca.fit_transform(X_st)

        pve = sklearn_pca.explained_variance_ratio_

        cum_pve = pve.sum()
        
        if verbose:
            print(numPCA,cum_pve)
        
        if cum_pve >= cum_pve_thresh:
            

            print('\n break \n')
            break


    print('numPCA', numPCA,'; cum_pve',cum_pve)

    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    PCA_df = pd.DataFrame(data = Y_pca, columns = pc_cols)


    return PCA_df, numPCA, cum_pve


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def findKmeansKopt(X,range_n_clusters,clusterMetric='silhScore',verbose=1):
    """
    Calculate kmeans, find optimal k number clusters.


    Parameters
    ----------
    X : numpy array
        MxN matrix of M instances and N features.
    range_n_clusters : range type
        Range of K for kmeans; minimum 2.
    clusterMetric : str, optional
        under construction: 'eucDist'.  The default is 'silhScore'.

    Returns
    -------
    Kopt : int 
        Optimal number of clusters.
    cluster_labels_best : list 
        Cluster labels.

    """

    

    metric_thresh = 0
    elbo_plot = []


    for n_clusters in range_n_clusters:

        print(f"kmeans on {n_clusters} clusters...")

        ### SciKit-Learn's Kmeans
        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        
        # #kmeans loss function
        # elbo_plot.append(kmeans.inertia_)        
        
        
        ####  Assign cluster labels
        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in kmeans.fit_predict(X)]

        
        
        if clusterMetric == 'silhScore':
            # Compute the silhouette scores for each sample
            silh_scores = silhouette_samples(X, cluster_labels)
            silh_scores_mean = np.mean(silh_scores)
            
            if verbose:
                print('max silh score:',np.max(silh_scores))
                print('min silh score:',np.min(silh_scores))            
                print('mean silh score:',silh_scores_mean)            
                
            
            if silh_scores_mean > metric_thresh:
                Kopt = n_clusters
                metric_thresh = silh_scores_mean
                cluster_labels_best = cluster_labels
                print('max mean silhouette score: ', silh_scores_mean)

            
#         elif clusterMetric == 'eucDist':
                # ... see selectKmeans()

    print(f"Best cluster: {Kopt}")


    return Kopt, cluster_labels_best, metric_thresh, elbo_plot
    
    
    
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
    
    

def selectKmeans(X,ev_IDs,Kopt,clusterMetric='silhScore'):
    """
    Calculate kmeans for select number of clusters.


    Parameters
    ----------
    X : numpy array
        MxN matrix of M instances and N features.
    ev_IDs : list
        List of event IDs for rows of X (to make dataframe for merging)        
    Kopt : int
        Number of clusters for kmeans
    clusterMetric : str, optional
        under construction: 'eucDist'.  The default is 'silhScore'.

    Returns
    -------
    Kopt_df : pandas dataframe
        columns are "ev_ID", "Cluster", "SS", "euc_dist"

    """

    

    

    n_clusters = Kopt


    ### SciKit-Learn's Kmeans
    kmeans = KMeans(n_clusters=n_clusters,
                       max_iter = 500,
                       init='k-means++', #how to choose init. centroid
                       n_init=10, #number of Kmeans runs
                       random_state=0) #set rand state

    
    
    ####  Assign cluster labels
    cluster_labels_0 = kmeans.fit_predict(X)
    #increment labels by one to match John's old kmeans code
    cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

    
    
    # Compute the silhouette scores for each sample
    silh_scores = silhouette_samples(X, cluster_labels)
    silh_scores_mean = np.mean(silh_scores)
    print('max mean silhouette score: ', silh_scores_mean)



    #get euclid dist to centroid for each point
    sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
    sum_sqr_dist = sqr_dist.sum(axis=1)
    euc_dist = np.sqrt(sum_sqr_dist)
            
    # #save centroids
    # centers = kmeans.cluster_centers_ 
    # #kmeans loss function
    # sse = kmeans.inertia_

    
    Kopt_df = pd.DataFrame(
              {'ev_ID':ev_IDs,
               'Cluster':cluster_labels,
               'SS':silh_scores,
               'euc_dist':euc_dist
               })




    return Kopt_df, silh_scores_mean


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def getTopFCat(cat0,topF,startInd=0,distMeasure = "SilhScore"):
    """


    Parameters
    ----------
    cat00 : all events
    topf : get top F events in each cluster
    startInd : can skip first event if needed
    Kopt : number of clusters
    distMeasure : type of distance emtrix between events. Default is "SilhScore",
    can also choose euclidean distance "EucDist"

    Returns
    -------
    catall : TYPE
        DESCRIPTION.

    """


    cat0['event_ID'] = [int(f) for f in  cat0['event_ID']]
    if distMeasure == "SilhScore":
        cat0 = cat0.sort_values(by='SS',ascending=False)

    if distMeasure == "EucDist":
        cat0 = cat0.sort_values(by='euc_dist',ascending=True)

    # try:
    cat0 = cat0[startInd:startInd+topF]
    # except: #if less than topF number events in cluster
    #     print(f"sampled all {len(cat0)} events in cluster!")

    # overwriting cat0 ?????
    return cat0




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def plotPCA(cat00,catall,Kopt, colors,size=5,size2=15, alpha=.5,labelpad = 5,fontsize=8,ax=None, fig=None):


    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()



    for k in range(1,Kopt+1):

        catk = cat00[cat00.Cluster == k]
        ax.scatter(catk.PC1,catk.PC3,catk.PC2,
                      s=size,
                      marker='x',
                      color=colors[k-1],
                      alpha=alpha)


# plot top FPs
    ax.scatter(catall.PC1,catall.PC3,catall.PC2,
                      s=size2,
                      marker='x',
                      color='k',
                      alpha=1)

    # sm = plt.cm.ScalarMappable(cmap=colors,
    #                            norm=plt.Normalize(vmin=df_stat.Cluster(),vmax=df_stat.Cluster()))
    axLabel = 'PC'#'Principal component '#label for plotting
    # cbar = plt.colorbar(sm,label=stat_name,shrink=.6,pad=.3);

    ax.set_xlabel(f'{axLabel} 1',labelpad=labelpad, fontsize = fontsize);
    ax.set_ylabel(f'{axLabel} 3',labelpad=labelpad, fontsize = fontsize);
    ax.set_zlabel(f'{axLabel} 2',labelpad=labelpad, fontsize = fontsize);
#     plt.colorbar(ticks=range(6), label='digit value')
#     plt.clim(-0.5, 5.5)

    ### Tick formatting is currently hard-coded
    # ax.set_xlim(-.6,.6)
    # ax.set_ylim(-.6,.6)
    # ax.set_zlim(-.6,.6)

    # ticks =  np.linspace(-.6,.6,5)
    # tick_labels = [f'{t:.1f}' for t in ticks]
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(tick_labels)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(tick_labels)
    # ax.set_zticks(ticks)
    # ax.set_zticklabels(tick_labels)


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



####FILTERs
##########################################################################################




def butter_bandpass(fmin, fmax, fs, order=5):
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fmin, fmax, fs, order=5):
    b, a = butter_bandpass(fmin, fmax, fs, order=order)
    y = lfilter(b, a, data)
    return y


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs):
    """
    Load waveform data from H5 file, apply 4th order bandpass filter,
    and zero mean

    Parameters
    ----------
    evID : int
    .
    dataH5_path : str
    .
    station : str
    .
    channel : str
    .
    fmin : int or float
    Minimum frequency for bandpass filter.
    fmax : int or float
    Maximum frequency for bandpass filter.
    tmin: 
    time (seconds) cut off beginning of record
    tmax:
    time (seconds) until end cut off (not including tmin)
    fs : int
    Sampling rate.

    Returns
    -------
    wf_zeromean : numpy array
    Filtered and zero-meaned waveform array.

    """

    with h5py.File(dataH5_path,'a') as fileLoad:

        wf_data_full = fileLoad[f'waveforms/{station}/{channel}'].get(str(evID))[:]

    wf_data = wf_data_full[int(tmin*fs):int(tmax*fs)]

    wf_filter = butter_bandpass_filter(wf_data, fmin,fmax,fs,order=4)
    wf_zeromean = wf_filter - np.mean(wf_filter)

    return wf_zeromean

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



def calcCCMatrix(catRep,shift_cc,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.
                    



    Returns np.array

    '''

    cc_mat = np.zeros([len(catRep),len(catRep)])
    lag_mat = np.zeros([len(catRep),len(catRep)])

    for i in range(len(catRep)):
        for j in range(len(catRep)):

            evIDA = catRep.event_ID.iloc[i]
            evIDB = catRep.event_ID.iloc[j]


            wf_A = getWF(evIDA,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)
            wf_B = getWF(evIDB,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)



            cc = correlate(wf_A, wf_B, shift_cc)
            lag, max_cc = xcorr_max(cc,abs_max=False)

            cc_mat[i,j] = max_cc
            lag_mat[i,j] = lag


    return cc_mat,lag_mat

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def calcCorr_template(wf_A,catRep,shift_cc,dataH5_path,station,channel,fmin,fmax,fs):
    ''' Calculate cross-correlation matrix and lag time for max CC coeg=f

    wf_A     : (np.array) wf template to match other waveforms

    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.


    Returns np.array
    '''


    cc_vec = np.zeros([len(catRep)]) #list cc coef

    lag_vec = np.zeros([len(catRep)]) #list lag time (samples) to get max cc coef


    for j in range(len(catRep)):

        evIDB = catRep.event_ID.iloc[j]

        wf_B = getWF(evIDB,dataH5_path,station,channel,fmin=fmin,fmax=fmax,fs=fs)



        cc = correlate(wf_A, wf_B, shift_cc)
        lag, max_cc = xcorr_max(cc)

        cc_vec[j] = max_cc
        lag_vec[j] = lag


    return cc_vec,lag_vec

# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def lagWF_Scalar(waveform, lag0):
    """
    Shift waveform amount to match template waveform. Pad to match original length. 


    Parameters
    ----------
    waveform : np.array

    lag0 : int
        Scalar from lag vector lag_vec, output of calcCorr_template.
    index_wf : int
        Index of waveform in catRep catalog.

    Returns
    -------
     np.array - time-shifted waveform to get maxCC

    """



    #
    if lag0<0:
#         print('neg lag', lag0[i])
        isNeg = 1
        lag00 = int(np.abs(lag0)) #convert to int
    else:
#         print('pos lag', lag0[i])
        isNeg = 0
        lag00 = int(lag0)


    padZ = np.zeros(lag00)# in samples
    padZ = np.ones(lag00)*np.nan# in samples

    if isNeg:
        waveform_shift = np.hstack([waveform,padZ])
        waveform_shift2 = waveform_shift[lag00:]

    else:
        waveform_shift = np.hstack([padZ,waveform])
        waveform_shift2 = waveform_shift[:-lag00]

    if lag0==0 or lag0==0.0:
        waveform_shift2 = waveform

    return waveform_shift2


def lagWF(waveform, lag0, index_wf):
    """
    Shift waveform amount to match template waveform.


    Parameters
    ----------
    waveform : np.array

    lag0 : np.array
        Matrix of lag times; output of calcCC_Mat.
    index_wf : int
        Index of waveform in catalog.

    Returns
    -------
     np.array - time-shifted waveform to get maxCC

    """


    i = index_wf

    #
    if lag0[i]<0:
#         print('neg lag', lag0[i])
        isNeg = 1
        lag00 = int(np.abs(lag0[i])) #convert to int
    else:
#         print('pos lag', lag0[i])
        isNeg = 0
        lag00 = int(lag0[i])


    padZ = np.zeros(lag00)# in samples
    padZ = np.ones(lag00)*np.nan# in samples

    if isNeg:
        waveform_shift = np.hstack([waveform,padZ])
        waveform_shift2 = waveform_shift[lag00:]

    else:
        waveform_shift = np.hstack([padZ,waveform])
        waveform_shift2 = waveform_shift[:-lag00]

    if lag0[i]==0 or lag0[i]==0.0:
        waveform_shift2 = waveform

    return waveform_shift2


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo






def swapLabels(cat,A,B):
    """
    Swap labels bewteen Cluster A and Cluster B.

    Parameters
    ----------
    cat : pd.DataFrame
        Must have column names: event_ID, Cluster.
    A : int, str
        Original cluster assignment.
    B : int, str
        New cluster assignment.

    Returns
    -------
    pd.DataFrame

    """



## swap label A to B
    dummy_variable = 999
    cat_swapped = cat.copy()
    cat_swapped.Cluster = cat_swapped.Cluster.replace(A,dummy_variable)
    cat_swapped.Cluster = cat_swapped.Cluster.replace(B,A)
    cat_swapped.Cluster = cat_swapped.Cluster.replace(dummy_variable,B)


    return cat_swapped




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo




def catMergeFromH5(path_Cat,path_proj,outfile_name):
    '''
    Keep csv catalog events based on H5 used in SpecUFEx

    '''

    ## read 'raw' catalog, the immutable one
    cat_raw = pd.read_csv(path_Cat)
    cat_raw['event_ID'] = [str(int(evv)) for evv in cat_raw['event_ID']]


    ## load event IDs from H5
    MLout =  h5py.File(path_proj + outfile_name,'r')
    evID_kept = [evID.decode('utf-8') for evID in MLout['catalog/event_ID/'][:]]
    MLout.close()

    ## put H5 events into pandas dataframe
    df_kept = pd.DataFrame({'event_ID':evID_kept})

    ## merge based on event ID
    cat00 = pd.merge(cat_raw,df_kept,on='event_ID')

    ## if length of H5 events and merged catalog are equal, then success
    if len(evID_kept) == len(cat00):
        print(f'{len(cat00)} events kept, merge sucessful')
    else:
        print('check merge -- error may have occurred ')


    ## convert to datetime, set as index
    cat00['datetime'] = [pd.to_datetime(i) for i in cat00.datetime]
    cat00['datetime_index']= [pd.to_datetime(i) for i in cat00.datetime]
    cat00 = cat00.set_index('datetime_index')


    return cat00


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo




def plotMapGMT(cat_feat,stat_df,buff,colorBy='depth_km',maxColor=10):


    cat_star = cat_feat[colorBy]

    region = [
        cat_feat.long.min() - buff,
        cat_feat.long.max() + buff,
        cat_feat.lat.min() - buff,
        cat_feat.lat.max() + buff,
    ]

    print(region)


    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="black", water="skyblue")

    if maxColor is not None:
        pygmt.makecpt(cmap="viridis", series=[cat_star.min(), maxColor])
    else:
        pygmt.makecpt(cmap="viridis", series=[cat_star.min(), cat_star.max()])

    fig.plot(
        x=cat_feat.long,
        y=cat_feat.lat,
        size=0.05 * 2 ** cat_feat.magnitude,
        color=cat_star,
        cmap=True,
        style="cc",
        pen="black",
    )

    if 'depth' in colorBy:
        fig.colorbar(frame='af+l"Depth (km)"')

    else:
        fig.colorbar(frame=f'af+l"{colorBy}"')

    fig.plot(x=stat_df.long, y=stat_df.lat,style="t.5c", color="pink", pen="black")




    fig.show()

    return fig




# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getFP(pathH5,evID):

    with h5py.File(pathH5,'r') as MLout:

        fp = MLout['fingerprints'].get(str(evID))[:]

        return fp
    
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




def getSgram(pathH5,evID):

    """
    Calculate cumulative percent variance explained for each principal component.

    Parameters
    ----------
    pathH5 : path to H5 with spectrograms
    evID   : event ID

    Returns
    -------
    specMat : arr
    tSTFT : arr
    fSTFT : arr

    """
    with h5py.File(pathH5,'r') as h5:


        specMat = h5['spectrograms'].get(str(evID))[:]
        
        tSTFT = h5['spec_parameters'].get('tSTFT')[()]
        fSTFT = h5['spec_parameters'].get('fSTFT')[()]

    return specMat, tSTFT, fSTFT






##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



import haversine


def latlon2meter(point1,point2):
    return haversine.haversine(point1, point2) * 1e3





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def calcMom(dmag):

    # INPUT:
    #
    # dmag       earthquake magnitude
    
    return 10**(1.2*dmag + 17); # moment, Bakun, 1984 for 1.5<ML<3.5






##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def calcRadius(magnitude,stressdrop=30,scale=1):
#       Matlab function to compute radius of constant stress drop
#       circular source given magnitude and stress drop
#
#       Input:
#               magnitude       earthquake magnitude
#               stress.drop     stress drop in bars
#       Output:
#               radius          source radius in meters

    return scale*(((7 * 10**(1.5 * magnitude + 16))/(16 * 1000000 * stressdrop))**(1/3))/(100);




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def upper_tri_masking(A):
    '''    
    #example
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(A)
    print(upper_tri_masking(A))    
    '''
    
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    
    return A[mask]





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def plotCircle(cat_clus,indd):
    
    
    mag = cat_clus.magnitude.iloc[indd]
    lat = cat_clus.lat.iloc[indd]
    lon = cat_clus.long.iloc[indd]
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r_m = calcRadius(mag)
    r = r_m / 111139 
    
    
    
    return (r)*np.cos(phi)+lon, (r)*np.sin(phi)+lat, lon, lat,r,r_m




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotCircleDepth(cat_clus,indd):
    
    
    mag = cat_clus.magnitude.iloc[indd]
    depth_m = cat_clus.depth_km.iloc[indd] * 1000
    lon = cat_clus.long.iloc[indd]
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r_m = calcRadius(mag)
    r = r_m / 111139 
    
    
    
    return (r)*np.cos(phi)+lon, (r_m)*np.sin(phi)+depth_m, lon, depth_m,r,r_m



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotErrorEllipse(cat_clus,indd):
    
    
    dX = cat_clus.dX.iloc[indd]
    dY = cat_clus.dY.iloc[indd]
    
    lon = cat_clus.long.iloc[indd]
    lat = cat_clus.lat.iloc[indd]
    
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    dX_deg = dX / 111139 
    dY_deg = dY / 111139     
    
    
    
    return (dX_deg)*np.cos(phi)+lon, (dY_deg)*np.sin(phi)+lat, lon, lat

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotErrorEllipseDepth(cat_clus,indd):
    
    
    dX = cat_clus.dX.iloc[indd]
    dZ = cat_clus.dZ.iloc[indd]
    
    lon = cat_clus.long.iloc[indd]
    depth_m = cat_clus.depth_km.iloc[indd]*1000
    
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    dX_deg = dX / 111139     
    
    
    return (dX_deg)*np.cos(phi)+lon, (dZ)*np.sin(phi)+depth_m, lon, depth_m

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotFPCombine(fp_df,cat_clust,numFP=2,numStates=15,fontsize=16,ax=None):

    '''
    fp_df = catalog of event fps where first (numFP*numStates) columns are the state transition probabilities
    numFP = how many FPs are we combining
    numStates = number of states in FP
    
    '''
    if ax is None:
        ax = plt.gca()


    repeater_clus = cat_clust.Cluster.iloc[0]
    fp_df_repeater = fp_df[fp_df.Cluster.isin([repeater_clus])]

    length_fp = numStates*numFP
    zeros_mat = np.zeros((length_fp,1))
    fp_concat = zeros_mat
    
    for i,evID in enumerate(fp_df_repeater.event_ID):
        
#         add matrix of zeros 
        
        #load and convert FP vector from catalog 
        fp = np.array(fp_df_repeater.iloc[i,range(length_fp*numStates)],dtype='float')

        #reshape to matrix
        fp_reshape = fp.reshape((length_fp,numStates))
        

        
        fp_concat = np.hstack([fp_concat,fp_reshape])       
        fp_concat = np.hstack([fp_concat,zeros_mat])               

        #plot
#         im = ax.imshow(fp_concat)
        plt.pcolormesh(fp_concat)
        ax = plt.gca()
#         cbar = plt.colorbar(im,pad=.06,shrink=.6,ax=ax)
#         cbar.set_label('Transition probability',labelpad=8)#,fontsize = 14)

# #         im.set_clim(0,1)


        labelsy = [str(r) for r in range(length_fp)]
        ticksy = range(length_fp)
        
        nclus = len(fp_df_repeater)
        labelsx = [str(r) for r in range(0,nclus+1)]
        ticksx = list(range(0,nclus*numStates+1,numStates))
                

        ax.set_ylabel('State')
        ax.set_yticks(ticksy)
        ax.set_yticklabels(labels=labelsy)
        
        ax.set_xlabel('Event')
        ax.set_xticks(ticksx)
        ax.set_xticklabels(labels=labelsx,rotation=90)        

#         ax.set_yticklabels([]) 
#         ax.set_ylabel('')    
#         ax.set_xticklabels([]) 
#         ax.set_xticks([])         
#         ax.set_yticks([])  
#         ax.set_xlabel('')    
    
    

    
    
    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################





def plotClusWF4_nolag(cat_clust,cat_spec_clus_trim,color,key_cluster,dataH5_path,station,channel,fmin,fmax,sampling_rate,alpha=.4,fontsize=10,offset=True,ax=None):
    ## filtered, time-trimmed
    
    if ax is None:
        ax = plt.gca()
    
    clus_id_list = list(cat_clust.event_ID)
    
    for j, evID in enumerate(cat_spec_clus_trim.event_ID):
        
        
        
        wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,sampling_rate)

        wf_norm = wf / np.max(np.abs(wf))
        wf_zeromean = wf_norm - np.mean(wf_norm)
        
#         if lag_index is not None:
#             wf_lag = lagWF(wf_zeromean, lag_mat[j,:], index_wf=lag_index)
#         else:
        wf_lag = wf_zeromean
        
        ## No offset waveforms
        if offset:
            offset = len(cat_clust)/15
            wf_offset = (wf_lag) + j*offset#1.5

        else:
            offset = 0
            wf_offset = (wf_lag)

        date = str(pd.Timestamp(cat_spec_clus_trim.timestamp.iloc[j]).date())
        r =  cat_spec_clus_trim.RID.iloc[j]
        ## new
        if r =='0' and evID in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='b') 
            if offset:
                ax.text(end_time_s,j*offset,s=f"{evID} - {date}",color='b',fontsize=14)            
        
        ##missing
        elif r =='0' and evID not in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha/2,c='k') 
            if offset:
                ax.text(end_time_s,j*offset,s=f"{evID} - {date}",color='k',fontsize=14)            
        
        ## NOT recovered 
        elif evID not in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha/2,c='k')              
            if offset:
                            
                ax.text(end_time_s,j*offset,s=f"{evID} - {date} - {r}",color='k',fontsize=14)            
                
                
        ## recovered 
        else:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='r')              
            if offset:
                            
                ax.text(end_time_s,j*offset,s=f"{evID} - {date} - {r}",color='r',fontsize=14)                
                        
        
        
        ax.set_ylabel('Normalized amplitude')    
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks(list(range(0,end_time_s+1,sampling_rate)))
        ax.set_xticklabels(list(range(0,int(np.ceil(end_time_s/sampling_rate))+1)))               
        ax.set_xlabel(f'Time (s)')   
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def plotClusWF4(cat_clust,cat_spec_clus_trim,lag_mat,lag_index,color,key_cluster,dataH5_path,station,channel,fmin,fmax,sampling_rate,alpha=.4,fontsize=10,offset=True,ax=None):
    ## filtered, time-trimmed
    
    if ax is None:
        ax = plt.gca()
    
    clus_id_list = list(cat_clust.event_ID)
    
    for j, evID in enumerate(cat_spec_clus_trim.event_ID):
        
        
        
        wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,sampling_rate)
        end_time_s = len(wf)#//sampling_rate

        wf_norm = wf / np.max(np.abs(wf))
        wf_zeromean = wf_norm - np.mean(wf_norm)
        
        if lag_index is not None:
            wf_lag = lagWF(wf_zeromean, lag_mat[j,:], index_wf=lag_index)
        else:
            wf_lag = wf_zeromean
        
        ## No offset waveforms
        if offset:
            offset = len(cat_clust)/15
            wf_offset = (wf_lag) + j*offset#1.5

        else:
            offset = 0
            wf_offset = (wf_lag)

        date = str(pd.Timestamp(cat_spec_clus_trim.timestamp.iloc[j]).date())
        r =  cat_spec_clus_trim.RID.iloc[j]
        ## new
        if r =='0' and evID in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='b') 
            if offset:
                ax.text(end_time_s,j*offset,s=f"{evID} - {date}",color='b',fontsize=14)            
        
        ##missing
        elif r =='0' and evID not in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha/2,c='k') 
            if offset:
                ax.text(end_time_s,j*offset,s=f"{evID} - {date}",color='k',fontsize=14)            
        
        ## NOT recovered 
        elif evID not in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha/2,c='k')              
            if offset:
                            
                ax.text(end_time_s,j*offset,s=f"{evID} - {date} - {r}",color='k',fontsize=14)            
                
                
        ## recovered 
        else:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='r')              
            if offset:
                            
                ax.text(end_time_s,j*offset,s=f"{evID} - {date} - {r}",color='r',fontsize=14)                
                        
        
        
        ax.set_ylabel('Normalized amplitude')    
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks(list(range(0,end_time_s+1,sampling_rate)))
        ax.set_xticklabels(list(range(0,int(np.ceil(end_time_s/sampling_rate))+1)))               
        ax.set_xlabel(f'Time (s)')   
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)






##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


    
def plotDepthZoom7(cat_clust,cat_spec_trim_missing,fontsize=16,ax=None):

    if ax is None:
        ax = plt.gca()
    ax.set_aspect(1/111139)
    
    ax.set_adjustable("datalim")
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    
    circ_scale = 1
    for indd in range(len(cat_spec_trim_missing)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_spec_trim_missing,indd)
        ax.plot(A*circ_scale,B*circ_scale, c='gray',ls='-' )
        ax.scatter(lon,depth_m, c='gray',ls='None',marker='x' )
        
        
    #### Missing, w/ RID
    cat_spec_trim_missing_RID = cat_spec_trim_missing[cat_spec_trim_missing.RID!='0']
    
    for indd in range(len(cat_spec_trim_missing_RID)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_spec_trim_missing,indd)

        ax.plot(A,B, c='lightsalmon',ls='-' )
        ax.scatter(lon,depth_m, c='lightsalmon',ls='None',marker='x' )        
    

    

    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_clust,indd)

        ax.plot(A*circ_scale,B*circ_scale, c='b',ls='-' )
        ax.scatter(lon,depth_m, c='b',ls='None',marker='x' )    

        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_clust_orig,indd)

        ax.plot(A*circ_scale,B*circ_scale, c='r',ls='-' )
        ax.scatter(lon,depth_m, c='r',ls='None',marker='x' )    
    
    #################################    
    #################################
    ### calculate Haversince distance
    
    
    buff = 0#.001        #0.0005
    buff_m = buff * 111139
    
    
#     point1y = (min(cat_clust.depth_km)-buff,np.median(cat_clust.long))
#     point2y = (max(cat_clust.depth_km)+buff,np.median(cat_clust.long))
    
    
    point1x = (np.median(cat_clust.lat),min(cat_clust.long))
    point2x = (np.median(cat_clust.lat),max(cat_clust.long))
    

    distx = int(np.ceil(latlon2meter(point1x,point2x)))    
#     disty = int(np.ceil(latlon2meter(point1y,point2y)))

    disty = max(cat_clust.depth_km) - min(cat_clust.depth_km)
    #################################
    #################################    
    
    
    
    ax.set_xlabel(f'EW epicentral \n distance = {distx} m',labelpad=20,fontsize=fontsize)
#     ax.set_ylabel(f'NS epicentral distance = {disty} m',labelpad=20,fontsize=fontsize)
    ax.set_ylabel(f'Depth (m)',labelpad=20,fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation = 45,size=8)   
    
    buff=.0005 
    buff_m = buff * 111139
    
    long_pair = (min(cat_clust.long)-buff,max(cat_clust.long)+buff)
    depth_pair = (1000*(min(cat_clust.depth_km)-buff_m),1000*(max(cat_clust.depth_km)+buff_m))
    
    depth_pair = (1000*min(cat_clust.depth_km)-buff_m,1000*max(cat_clust.depth_km)+buff_m)
    
    
    ax.set_xticks([])
    ax.set_xticklabels([])  
    ax.set_xlim(long_pair)
    ax.set_ylim(depth_pair)

    ax.invert_yaxis()
    ax.grid('on')
#     print(depth_pair)

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    
def plotDepthZoom8(cat_clust,cat_spec_trim_missing,fontsize=16,buff=.0005,ax=None):

    if ax is None:
        ax = plt.gca()
    ax.set_aspect(1/111139)
    
    ax.set_adjustable("datalim")
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    
    circ_scale = 1
    for indd in range(len(cat_spec_trim_missing)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_spec_trim_missing,indd)
        ax.plot(A*circ_scale,B*circ_scale, c='gray',ls='-' )
        ax.scatter(lon,depth_m, c='gray',ls='None',marker='x' )
        
        
    #### Missing, w/ RID
    cat_spec_trim_missing_RID = cat_spec_trim_missing[cat_spec_trim_missing.RID!='0']
    
    for indd in range(len(cat_spec_trim_missing_RID)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_spec_trim_missing,indd)

        ax.plot(A,B, c='lightsalmon',ls='-' )
        ax.scatter(lon,depth_m, c='lightsalmon',ls='None',marker='x' )        
    

    

    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_clust,indd)

        ax.plot(A*circ_scale,B*circ_scale, c='r',ls='-' )
        ax.scatter(lon,depth_m, c='r',ls='None',marker='x' )    

        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A, B, lon, depth_m,r, r_m = plotCircleDepth(cat_clust_orig,indd)

        ax.plot(A*circ_scale,B*circ_scale, c='k',ls='-' )
        ax.scatter(lon,depth_m, c='k',ls='None',marker='x' )    
    
    #################################    
    #################################
    ### calculate Haversince distance
    
    
    buff = 0#.001        #0.0005
    buff_m = buff * 111139
    
    
#     point1y = (min(cat_clust.depth_km)-buff,np.median(cat_clust.long))
#     point2y = (max(cat_clust.depth_km)+buff,np.median(cat_clust.long))
    
    
    point1x = (np.median(cat_clust.lat),min(cat_clust.long))
    point2x = (np.median(cat_clust.lat),max(cat_clust.long))
    

    distx = int(np.ceil(latlon2meter(point1x,point2x)))    
#     disty = int(np.ceil(latlon2meter(point1y,point2y)))

    disty = max(cat_clust.depth_km) - min(cat_clust.depth_km)
    #################################
    #################################    
    
    
    
    ax.set_xlabel(f'EW epicentral \n distance = {distx} m',labelpad=20,fontsize=fontsize)
#     ax.set_ylabel(f'NS epicentral distance = {disty} m',labelpad=20,fontsize=fontsize)
    ax.set_ylabel(f'Depth (m)',labelpad=20,fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation = 45,size=8)   
    
    buff=.0005 
    buff_m = buff * 111139
    
    long_pair = (min(cat_clust.long)-buff,max(cat_clust.long)+buff)
    depth_pair = (1000*(min(cat_clust.depth_km)-buff_m),1000*(max(cat_clust.depth_km)+buff_m))
    
    depth_pair = (1000*min(cat_clust.depth_km)-buff_m,1000*max(cat_clust.depth_km)+buff_m)
    
    
    ax.set_xticks([])
    ax.set_xticklabels([])  
    ax.set_xlim(long_pair)
    ax.set_ylim(depth_pair)

    ax.invert_yaxis()
    ax.grid('on')
#     print(depth_pair)
# 
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def plotCircleDepthMeters(cat_clus,indd):
    
    
    mag = cat_clus.magnitude.iloc[indd]
    X = cat_clus.X.iloc[indd]
    Z = cat_clus.Z.iloc[indd]
    
    dX = cat_clus.dX.iloc[indd]
    dZ = cat_clus.dZ.iloc[indd]
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r_m = calcRadius(mag)
        
    
    return (r_m)*np.cos(phi)+X, (r_m)*np.sin(phi)+Z, X, Z, dX,dZ,r_m

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


    
def plotDepthZoomMeters(cat_clust,cat_spec_trim_missing,fontsize=16,ax=None):

    if ax is None:
        ax = plt.gca()
       
#     ax.set_aspect(1/111139)
    
#     ax.set_adjustable("datalim")
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    

    for indd in range(len(cat_spec_trim_missing)):

        A2, B2, X, Z, dX,dZ, r_m =  plotCircleDepthMeters(cat_spec_trim_missing,indd)

        ax.plot(A2,B2, c='lightsalmon',ls='-' )

        dX = max(10,dX)
        dZ = max(10,dZ)
        ###error bars    
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')
        
        
    #### Missing, w/ RID
    cat_spec_trim_missing_RID = cat_spec_trim_missing[cat_spec_trim_missing.RID!='0']
    
    for indd in range(len(cat_spec_trim_missing_RID)):

        A2, B2, X, Z, dX,dZ, r_m = plotCircleDepthMeters(cat_spec_trim_missing_RID,indd)

        ax.plot(A2,B2, c='lightsalmon',ls='-' )
        
        dX = max(10,dX)
        dZ = max(10,dZ)
        ###error bars    
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')


    

    

    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A2, B2, X, Z, dX,dZ, r_m = plotCircleDepthMeters(cat_clust,indd)

        ax.plot(A2,B2, c='r',ls='-' )
        
        dX = max(10,dX)
        dZ = max(10,dZ)
        ###error bars    
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')



        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A2, B2, X, Z, dX,dZ, r_m = plotCircleDepthMeters(cat_clust_orig,indd)

        ax.plot(A2,B2, c='b',ls='-' )

        ###error bars 
        dX = max(10,dX)
        dZ = max(10,dZ)        
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')


        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)') 
    

    
    
    ax.set_xlabel(f'X m',labelpad=20,fontsize=fontsize)
    ax.set_ylabel(f'Z (m)',labelpad=20,fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation = 45,size=8)   
    


    ax.invert_yaxis()
    ax.grid('on')
#     print(depth_pair)
# 
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




        
        
def plotCCMat2(cc_mat,cat_clust,fontsize=12,colorbar=True,cb_pad=.06,cb_shrink=.6,cb_or='vertical',ax=None):

    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(cc_mat)
    
    if colorbar:
        cbar = plt.colorbar(im,pad=cb_pad,shrink=cb_shrink,orientation=cb_or,ax=ax)
        cbar.set_label('Correlation coefficient',labelpad=8)#,fontsize = 14)
        im.set_clim(0,1)

    dates = [pd.to_datetime(d).date() for d in cat_clust.index]
    
    labelsy = dates
    ticksy = range(len(dates))

    labels = list(cat_clust.event_ID)
    ticks = range(len(cat_clust.event_ID))    
    
    ax.set_xlabel('Event ID')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels=labels,rotation=90,fontsize=fontsize)

    ax.set_yticks(ticksy)
    ax.set_yticklabels(labelsy,fontsize=fontsize) 
    ax.set_ylabel('Date') 
    




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotStem3(cat_clust,cat_spec_clus_trim_missing,RIDs,R,cat_full,fontsize=2,ax=None):
    """
    RIDs: list of all RIDS from 2014 catalog
    R: RID of specific sequence of interest
    
    """

    if ax is None:
        ax = plt.gca()
        
        
        
        
    #Get missing wf
    cumsum_missing = [i for i in range(0,len(cat_spec_clus_trim_missing))]
    mags_missing = [m for m in cat_spec_clus_trim_missing.magnitude] 
    mags_s_missing = [1.5*10**m for m in cat_spec_clus_trim_missing.magnitude]    
    
    dates_missing = [pd.to_datetime(d) for d in cat_spec_clus_trim_missing.index]
    labels_missing = [str(d).split(' ')[0] for d in dates_missing]        
    
           

    ## Fingerprints in cluster
    cat_clust = cat_clust.sort_index()
    cumsum = [i for i in range(0,len(cat_clust))]
    mags = [m for m in cat_clust.magnitude]
    mags_s = [1.5*10**m for m in cat_clust.magnitude]
    
    dates = [pd.to_datetime(d) for d in cat_clust.index]
    labels = [str(d).split(' ')[0] for d in dates]    

    
    #Get already cataloged REQS
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    cumsum_orig = [i for i in range(0,len(cat_clust_orig))]
    mags_orig = [m for m in cat_clust_orig.magnitude] 
    mags_s_orig = [1.5*10**m for m in cat_clust_orig.magnitude]    
    
    dates_orig = [pd.to_datetime(d) for d in cat_clust_orig.index]
    labels_orig = [str(d).split(' ')[0] for d in dates_orig]    
    
    

    try:
        markerline, stemline, baseline, = ax.stem(dates_missing,mags_missing,mags_s_missing,markerfmt='k*',linefmt='k-',basefmt='None')
    except:
        pass 

    markerline, stemline, baseline, = ax.stem(dates, mags,mags_s,markerfmt='b*',linefmt='b-',basefmt='None')

    try:
        markerline, stemline, baseline, = ax.stem(dates_orig,mags_orig,mags_s_orig,markerfmt='r*',linefmt='r-',basefmt='None')
    except:
        pass

    
    dates_all = list(set(dates + dates_missing))
    dates_all.sort()
    
    labels_all = list(set(labels + labels_missing))    
    labels_all.sort()
    
    
    try:
        ax.set_xticks(dates_all)
        ax.set_xticklabels(labels_all,fontsize=fontsize)    
        
    except:
        pass
    
    ax.tick_params(axis='x', labelrotation = 90,size=fontsize)       
    ax.set_title(R,fontsize=20)
    
    

    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Date')
    





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotStem4(cat_clust,RIDs,R,cat_full,fontsize=2,ax=None):
    """
    RIDs: list of all RIDS from 2014 catalog
    R: RID of specific sequence of interest
    
    """

    if ax is None:
        ax = plt.gca()
        
        
           

    ## Fingerprints in cluster
    cat_clust = cat_clust.sort_index()
    cumsum = [i for i in range(0,len(cat_clust))]
    mags = [m for m in cat_clust.magnitude]
    mags_s = [1.5*10**m for m in cat_clust.magnitude]
    
    dates = [pd.to_datetime(d).year for d in cat_clust.index]
    labels = [str(d).split(' ')[0] for d in dates]
    try:
        nStat =  [str(int(n)) for n in cat_clust.NumStations]
    except:
        pass
    
    #Get already cataloged REQS
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    cumsum_orig = [i for i in range(0,len(cat_clust_orig))]
    mags_orig = [m for m in cat_clust_orig.magnitude] 
    mags_s_orig = [1.5*10**m for m in cat_clust_orig.magnitude]    
    
    dates_orig = [pd.to_datetime(d) for d in cat_clust_orig.index]
    labels_orig = [str(d).split(' ')[0] for d in dates_orig]    
    
    

    markerline, stemline, baseline, = ax.stem(dates, mags,mags_s,markerfmt='b*',linefmt='b-',basefmt='None')

    try:
        markerline, stemline, baseline, = ax.stem(dates_orig,mags_orig,mags_s_orig,markerfmt='r*',linefmt='r-',basefmt='None')
    except:
        pass

    
    dates_all = list(set(dates))
    dates_all.sort()
    
    labels_all = list(set(labels))    
    labels_all.sort()

   
    
    
    ax.set_xticks(dates_all)
    try:
        ax.set_xticklabels(labels_all,fontsize=fontsize) 
    except:
        pass
    
    try:
        for d,m,n in zip(dates,mags,nStat):
            ax.text(d,m+.1,n)
    except:
        pass
    
    ax.tick_params(axis='x', labelrotation = 90,size=fontsize)       
    ax.set_title(R,fontsize=20)
    
    

    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Date')
    ax.set_ylim(0,max(mags)+.1*max(mags))
    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def plotStem5(cat_clust,RIDs,R,cat_full,fontsize=2,ax=None):
    """
    RIDs: list of all RIDS from 2014 catalog
    R: RID of specific sequence of interest
    
    """

    if ax is None:
        ax = plt.gca()
        
        
           

    ## Fingerprints in cluster
    cat_clust = cat_clust.sort_index()
    cumsum = [i for i in range(0,len(cat_clust))]
    mags = [m for m in cat_clust.magnitude]
    mags_s = [1.5*10**m for m in cat_clust.magnitude]
    
    dates = [pd.to_datetime(d).year for d in cat_clust.index]
    labels = [str(d).split(' ')[0] for d in dates]
    try:
        nStat =  [str(int(n)) for n in cat_clust.NumStations]
    except:
        pass
    
    #Get already cataloged REQS
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    cumsum_orig = [i for i in range(0,len(cat_clust_orig))]
    mags_orig = [m for m in cat_clust_orig.magnitude] 
    mags_s_orig = [1.5*10**m for m in cat_clust_orig.magnitude]    
    
    dates_orig = [pd.to_datetime(d).year for d in cat_clust_orig.index]
    labels_orig = [str(d).split(' ')[0] for d in dates_orig]    
    
    

    markerline, stemline, baseline, = ax.stem(dates, mags,mags_s,markerfmt='r*',linefmt='r-',basefmt='None')

    try:
        markerline, stemline, baseline, = ax.stem(dates_orig,mags_orig,mags_s_orig,markerfmt='b*',linefmt='b-',basefmt='None')
    except:
        pass

    
    dates_all = list(set(dates))
    dates_all.sort()
    
    labels_all = list(set(labels))    
    labels_all.sort()

   
    
    
    ax.set_xticks(dates_all)
    try:
        ax.set_xticklabels(labels_all,fontsize=fontsize) 
    except:
        pass
    
#     try:
#         for d,m,n in zip(dates,mags,nStat):
#             ax.text(d,m+.1,n)
#     except:
#         pass
    
    ax.tick_params(axis='x', labelrotation = 90,size=fontsize)       
    ax.set_title(R,fontsize=20)
    
    

    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Date')
    ax.set_ylim(0,max(mags)+.1*max(mags))
    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




def plotStem7(cat_clust,fontsize=12,ax=None):
    """
    RIDs: list of all RIDS from 2014 catalog
    R: RID of specific sequence of interest
    
    """

    if ax is None:
        ax = plt.gca()
        
        
           

    ## Fingerprints in cluster
    cat_clust = cat_clust.sort_index()
    cumsum = [i for i in range(0,len(cat_clust))]
    mags = [m for m in cat_clust.magnitude]
    mags_s = [1.5*10**m for m in cat_clust.magnitude]
    
    dates = [pd.to_datetime(d).year for d in cat_clust.index]
    labels = [str(d).split(' ')[0] for d in dates]
    try:
        nStat =  [str(int(n)) for n in cat_clust.NumStations]
    except:
        pass
    
    #Get already cataloged REQS
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    cumsum_orig = [i for i in range(0,len(cat_clust_orig))]
    mags_orig = [m for m in cat_clust_orig.magnitude] 
    mags_s_orig = [1.5*10**m for m in cat_clust_orig.magnitude]    
    
    dates_orig = [pd.to_datetime(d).year for d in cat_clust_orig.index]
    labels_orig = [str(d).split(' ')[0] for d in dates_orig]    
    
    



    markerline, stemline, baseline, = ax.stem(dates_orig,mags_orig,mags_s_orig,markerfmt='k*',linefmt='k-',basefmt='None')

    
    dates_all = list(set(dates))
    dates_all.sort()
    
    labels_all = list(set(labels))    
    labels_all.sort()

   
    
    
    ax.set_xticks(dates_all)
    try:
        ax.set_xticklabels(labels_all,fontsize=fontsize,rotation=45) 
    except:
        pass
    
    try:
        for d,m,n in zip(dates,mags,nStat):
            ax.text(d,m+.1,n)
    except:
        pass
    
    ax.tick_params(axis='x', labelrotation = 90,size=fontsize,rotation=45)           
    

    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Date')
    ax.set_ylim(0,max(mags)+.1*max(mags))
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotLocZoom4(cat_clust,cat_full,fontsize=16,ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_aspect('equal')
    
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    
    
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###   
    ## Get area of hypocenters 
    mag_avg = np.mean(cat_clust.magnitude)  
    
    r_m = calcRadius(mag_avg)
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r = r_m / 111139 
    
    maxlat  = max(cat_clust.lat)+r
    maxlong = max(cat_clust.long)+r
    minlat  = min(cat_clust.lat)-r
    minlong = min(cat_clust.long)-r

    cat_spec_clus_trim = cat_full[cat_full.long<=maxlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat<=maxlat]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.long>=minlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat>=minlat]
    
    
    for indd in range(len(cat_spec_clus_trim)):

        A, B, lon, lat,r, r_m = plotCircle(cat_spec_clus_trim,indd)

        ax.plot(A,B, c='gray',ls='-' )
        ax.scatter(lon,lat, c='gray',ls='None',marker='x' )
    

    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A, B, lon, lat,r, r_m = plotCircle(cat_clust,indd)

        ax.plot(A,B, c='b',ls='-' )
        ax.scatter(lon,lat, c='b',ls='None',marker='x' )    

        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A, B, lon, lat,r, r_m = plotCircle(cat_clust_orig,indd)

        ax.plot(A,B, c='r',ls='-' )
        ax.scatter(lon,lat, c='r',ls='None',marker='x' )    
    
    #################################    
    #################################
    ### calculate Haversince distance
    
    
    buff = 0#.001        #0.0005
    buff_m = buff * 111139
    
    
    point1y = (min(cat_clust.lat)-buff,np.median(cat_clust.long))
    point2y = (max(cat_clust.lat)+buff,np.median(cat_clust.long))
    
    
    point1x = (np.median(cat_clust.lat),min(cat_clust.long)-buff)
    point2x = (np.median(cat_clust.lat),max(cat_clust.long)+buff)
    

    distx = int(np.ceil(latlon2meter(point1x,point2x)))    
    disty = int(np.ceil(latlon2meter(point1y,point2y)))    
    #################################
    #################################    
    
    
    
    ax.set_xlabel(f'EW epicentral \n Distance = {distx} m',labelpad=20,fontsize=fontsize)
    ax.set_ylabel(f'NS epicentral \n Distance = {disty} m',labelpad=20,fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation = 45,size=8)   
    
    buff=.001 
    long_pair = (min(cat_clust.long)-buff,max(cat_clust.long)+buff)
    lat_pair = (min(cat_clust.lat)-buff,max(cat_clust.lat)+buff)

        
    ax.set_xlim(long_pair)
    ax.set_ylim(lat_pair)
    
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])   
    ax.set_xticks([])
    ax.set_yticks([]) 
    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def plotLocZoom5(cat_clust,cat_full,fontsize=16, buff=.001, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_aspect('equal')
    
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    
    
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###   
    ## Get area of hypocenters 
    mag_avg = np.mean(cat_clust.magnitude)  
    
    r_m = calcRadius(mag_avg)
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r = r_m / 111139 
    
    maxlat  = max(cat_clust.lat)+r
    maxlong = max(cat_clust.long)+r
    minlat  = min(cat_clust.lat)-r
    minlong = min(cat_clust.long)-r

    cat_spec_clus_trim = cat_full[cat_full.long<=maxlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat<=maxlat]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.long>=minlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat>=minlat]
    
    
    for indd in range(len(cat_spec_clus_trim)):

        A, B, lon, lat,r, r_m = plotCircle(cat_spec_clus_trim,indd)

        ax.plot(A,B, c='gray',ls='-' )
        ax.scatter(lon,lat, c='gray',ls='None',marker='x' )
    

    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A, B, lon, lat,r, r_m = plotCircle(cat_clust,indd)

        ax.plot(A,B, c='r',ls='-' )
        ax.scatter(lon,lat, c='r',ls='None',marker='x' )    

        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red### 
    

    for indd in range(len(cat_clust_orig)):

        A, B, lon, lat,r, r_m = plotCircle(cat_clust_orig,indd)

        ax.plot(A,B, c='k',ls='-' )
        ax.scatter(lon,lat, c='k',ls='None',marker='x' )    
    
    #################################    
    #################################
    ### calculate Haversince distance
    

    point1y = (min(cat_clust.lat),np.median(cat_clust.long))
    point2y = (max(cat_clust.lat),np.median(cat_clust.long))
    
    
    point1x = (np.median(cat_clust.lat),min(cat_clust.long))
    point2x = (np.median(cat_clust.lat),max(cat_clust.long))
    

    distx = int(np.ceil(latlon2meter(point1x,point2x)))    
    disty = int(np.ceil(latlon2meter(point1y,point2y)))    
    #################################
    #################################    
    
    
    
    ax.set_xlabel(f'EW epicentral \n distance = {distx} m',labelpad=20,fontsize=fontsize)
    ax.set_ylabel(f'NS epicentral \n distance = {disty} m',labelpad=20,fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation = 45,size=8)   
    
    long_pair = (min(cat_clust.long)-buff,max(cat_clust.long)+buff)
    lat_pair = (min(cat_clust.lat)-buff,max(cat_clust.lat)+buff)

        
    ax.set_xlim(long_pair)
    ax.set_ylim(lat_pair)
    
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])   
    ax.set_xticks([])
    ax.set_yticks([]) 
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotCircleMeters(cat_clus,indd):
    
    
    mag = cat_clus.magnitude.iloc[indd]
    X = cat_clus.X.iloc[indd]
    Y = cat_clus.Y.iloc[indd]
    
    dX = cat_clus.dX.iloc[indd]
    dY = cat_clus.dY.iloc[indd]
    
    
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r_m = calcRadius(mag)

    
    
    return (r_m)*np.cos(phi)+X, (r_m)*np.sin(phi)+Y, X, Y, dX,dY,r_m
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotCircleDepthMeters(cat_clus,indd):
    
    
    mag = cat_clus.magnitude.iloc[indd]
    X = cat_clus.X.iloc[indd]
    Z = cat_clus.Z.iloc[indd]
    
    dX = cat_clus.dX.iloc[indd]
    dZ = cat_clus.dZ.iloc[indd]
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r_m = calcRadius(mag)
        
    
    return (r_m)*np.cos(phi)+X, (r_m)*np.sin(phi)+Z, X, Z, dX,dZ,r_m


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotLocZoomMeters(cat_clust,cat_full,fontsize=16,ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_aspect('equal')
    
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    
    
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###   
    ## Get area of hypocenters 
    mag_avg = np.mean(cat_clust.magnitude)  
    
    r_m = calcRadius(mag_avg)
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r = r_m / 111139 
    
    maxlat  = max(cat_clust.lat)+r
    maxlong = max(cat_clust.long)+r
    minlat  = min(cat_clust.lat)-r
    minlong = min(cat_clust.long)-r

    cat_spec_clus_trim = cat_full[cat_full.long<=maxlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat<=maxlat]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.long>=minlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat>=minlat]
    

    for indd in range(len(cat_spec_clus_trim)):

        A2, B2, X, Y, dX,dY, r_m = plotCircleMeters(cat_spec_clus_trim,indd)

        ax.plot(A2,B2, c='grey',ls='-' )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)') 

        dX = max(10,dX)
        dY = max(10,dY)
        ##plot error pars
        ax.scatter(X,Y, c='gray',ls='None',marker='x' )    
        ax.vlines(X,Y-dY, Y+dY, linestyles='-', colors='gray')
        ax.hlines(Y,X-dX, X+dX, linestyles='-', colors='gray')


    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A2, B2, X, Y, dX,dY, r_m = plotCircleMeters(cat_clust,indd)

        ax.plot(A2,B2, c='r',ls='-' )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)') 

        dX = max(10,dX)
        dY = max(10,dY)
        ##plot error pars
        ax.scatter(X,Y, c='gray',ls='None',marker='x' )    
        ax.vlines(X,Y-dY, Y+dY, linestyles='-', colors='gray')
        ax.hlines(Y,X-dX, X+dX, linestyles='-', colors='gray')

        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A2, B2, X, Y, dX,dY, r_m = plotCircleMeters(cat_clust_orig,indd)

        ax.plot(A2,B2, c='b',ls='-' )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)') 

        dX = max(10,dX)
        dY = max(10,dY)
        ##plot error pars
        ax.scatter(X,Y, c='gray',ls='None',marker='x' )    
        ax.vlines(X,Y-dY, Y+dY, linestyles='-', colors='gray')
        ax.hlines(Y,X-dX, X+dX, linestyles='-', colors='gray')

    ax.grid('on')

    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################







##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
def calcCCMatrixFP(catRep,SpecUFEx_H5_path):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.
                    



    Returns np.array

    '''

    cc_mat = np.zeros([len(catRep),len(catRep)])
    lag_mat = np.zeros([len(catRep),len(catRep)])

    for i in range(len(catRep)):
        for j in range(len(catRep)):

            evIDA = catRep.event_ID.iloc[i]
            evIDB = catRep.event_ID.iloc[j]
            
            
            #linearize fingerprints
            with h5py.File(SpecUFEx_H5_path,'r') as MLout:
                fpA = MLout['fingerprints'].get(evIDA)[:]                
                fpB = MLout['fingerprints'].get(evIDB)[:]

                wf_A = fpA.reshape(1,len(fpA)**2)[:][0]     
                wf_B = fpB.reshape(1,len(fpB)**2)[:][0]            
                

            cc = correlate(wf_A, wf_B, 0)
            lag, max_cc = xcorr_max(cc,abs_max=False)#Determines if the largest value of the correlation function is returned, independent of it being positive (correlation) or negative (anti-correlation). If False the maximum returned is positive only.

            cc_mat[i,j] = max_cc
            lag_mat[i,j] = lag


    return cc_mat,lag_mat# 




#oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
def calcEucMatrixFP(catRep,SpecUFEx_H5_path):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    Returns np.array

    '''

    ed_mat = np.zeros([len(catRep),len(catRep)])

    for i in range(len(catRep)):
        for j in range(len(catRep)):

            evIDA = catRep.event_ID.iloc[i]
            evIDB = catRep.event_ID.iloc[j]
            
            
            #linearize fingerprints
            with h5py.File(SpecUFEx_H5_path,'r') as MLout:
                fpA = MLout['fingerprints'].get(evIDA)[:]                
                fpB = MLout['fingerprints'].get(evIDB)[:]

                fpA = fpA.reshape(1,len(fpA)**2)[:][0]     
                fpB = fpB.reshape(1,len(fpB)**2)[:][0]            
                

            ed = euclidean(fpA, fpB)

            ed_mat[i,j] = ed



    return ed_mat
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
from matplotlib.colorbar import cm

def plotSgram(specMat,evID,tSTFT, fSTFT,ax=None):
    #set x to datetime list or seconds in tSTFT
    # x=tSTFT
    if ax is None:
        ax = plt.gca()

    plt.pcolormesh(tSTFT, fSTFT, specMat,cmap=cm.magma, shading='auto')

    cbar = plt.colorbar(pad=.06)
    cbar.set_label('dB',labelpad=8)#,fontsize = 14)
#     plt.clim(0,45)
    date_title = str(pd.to_datetime('200' + evID))
    ax.set_title(date_title,pad=10)






    ax.set_ylabel('f (Hz)',labelpad=10)
    ax.set_xlabel('t (s)')

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotFP(fp, ax=None):
    
    if ax is None:
        ax = plt.gca()
        
    
    numStates = fp.shape[0]
    
    ax.imshow(fp)
    
    ax.set_xlabel("Current state")
    ax.set_ylabel("Next state")   
    
    ax.set_xticks(range(0,numStates,5))
    ax.set_yticks(range(0,numStates,5)) 
    ax.invert_yaxis()
    


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotWF(evID,dataH5_path,station,channel,fmin,fmax,fs,tSTFT,colorBy='cluster',k='',ax=None,**plt_kwargs):
    '''
    '''

    colors      =     plt_kwargs['colors']

    if ax is None:
        ax = plt.gca()


    wf_zeromean = getWF(evID,dataH5_path,station,channel,fmin,fmax,fs)


    if colorBy=='cluster':

        colorWF = colors[k-1]
    else:
        colorWF = 'k'

    ax.plot(wf_zeromean,color=colorWF,lw=1)

    plt.ylabel('Velocity')



# #### General
    ticks=[np.floor(c) for c in np.linspace(0,len(wf_zeromean),3)]
    ticklabels=[f'{c:.0f}' for c in np.linspace(0,np.ceil(max(tSTFT)),3)]
    plt.xticks(ticks=ticks,labels=ticklabels)

    plt.xlabel('t (s)')
    plt.xlim(0,len(wf_zeromean))
    return wf_zeromean








##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################






def getFP(pathH5,evID):

    with h5py.File(pathH5,'r') as MLout:

        fp = MLout['fingerprints'].get(str(evID))[:]

        return fp
    
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotCumul2(cat_clust,color1,color2,ax=None):
    
    if ax is None:
        ax = plt.gca()
        
    
    cat_clust = cat_clust.sort_index()   
    cumsum = [i for i in range(0,len(cat_clust))]
    dates = [pd.to_datetime(d) for d in cat_clust.index]
    labels = [str(d).split(' ')[0] for d in dates]
    
    ### calc slope###
    ### calc slope###
    ### calc slope###
    x = [np.datetime64(p).astype(int) for p in dates]
    y = cumsum

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    def linFunc(x):
        return slope * x + intercept

    linModel = list(map(linFunc, x))
    dy = linModel[-1]-linModel[0]
    dx = (dates[-1]-dates[0]).days / 365.25
    
    slope2 = dy/dx
    
    ax.plot(dates, linModel,color=color2,alpha=1,lw=2,ls='--')
    ax.scatter(dates, cumsum,c=color1,s=1)
    
        
    ax.set_ylabel('Cumalitive # earthquakes')    

    ax.tick_params(axis='x', labelrotation = 90,size=10)        
    
    ax.grid('on')
    
    return slope2, r, dates
    

from scipy import stats

def plotCumul3(cat_clust,color1,color2,size=1,ax=None):
    
    if ax is None:
        ax = plt.gca()
        
    
    cat_clust = cat_clust.sort_index()   
    cumsum = [i for i in range(0,len(cat_clust))]
    dates = [pd.to_datetime(d) for d in cat_clust.index]
    labels = [str(d).split(' ')[0] for d in dates]
    
    ### calc slope###
    ### calc slope###
    ### calc slope###
    x = [np.datetime64(p).astype(int) for p in dates]
    y = cumsum

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    def linFunc(x):
        return slope * x + intercept

    linModel = list(map(linFunc, x))
    dy = linModel[-1]-linModel[0]
    dx = (dates[-1]-dates[0]).days / 365.25
    
    slope2 = dy/dx
    
#     ax.plot(dates, linModel,color=color2,alpha=1,lw=2,ls='--')
    ax.scatter(dates, cumsum,c=color1,s=size)
    
        
    ax.set_ylabel('Cumalitive # earthquakes')    

    ax.tick_params(axis='x', labelrotation = 90,size=10)        
    
    ax.grid('on')
    
    return slope2, r, dates
    

def plotCumul4(cat_clust,size=50,ax=None):
    
    if ax is None:
        ax = plt.gca()
        
    
    cat_clust = cat_clust.sort_index()   
    cumsum = [i for i in range(0,len(cat_clust))]
    dates = [pd.to_datetime(d) for d in cat_clust.index]    
    
    cat_clust_old = cat_clust[cat_clust.RID!='0'].sort_index()   
    cumsum_old = [i for i in range(0,len(cat_clust_old))]
    dates_old = [pd.to_datetime(d) for d in cat_clust_old.index]

    
    cat_clust_new = cat_clust[cat_clust.RID=='0'].sort_index()   
    cumsum_new = [i for i in range(0,len(cat_clust_new))]
    dates_new = [pd.to_datetime(d) for d in cat_clust_new.index]

    ax.step(dates, cumsum,color='grey',where='post')
    ax.scatter(dates, cumsum,c='grey',s=size)
        
    ax.set_ylabel('Cumalitive number of earthquakes')    

    ax.tick_params(axis='x', labelrotation = 90,size=10)        
    
    ax.grid('on')
    
    




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotCumulMoment(cat_clust,color1='darkorange',color2='orange',ms=1,ax=None):
    
    if ax is None:
        ax = plt.gca()
        
    
    cat_clust = cat_clust.sort_index()   
    cumsum = [i for i in np.cumsum(cat_clust.moment)]
    mags = [i for i in cat_clust.magnitude]

    dates = [pd.to_datetime(d) for d in cat_clust.index]
    
    labels = [str(d).split(' ')[0] for d in dates]
#     ax2 = ax.twinx()
    
#     ax2.stem(dates, mags)
    
    ### calc slope###
    ### calc slope###
    ### calc slope###
#     ax.plot(dates, linModel,color=color2,alpha=1,lw=2,ls='--')
    ax.plot(dates, cumsum,c=color1,lw=2,ls='None',marker='o',ms=ms)
    
    ax.spines['right'].set_color(color1)
    ax.yaxis.label.set_color(color1)
    ax.tick_params(axis='y', colors=color1)
    
#     ax.stem(dates, cumsum)#,c=color1,lw=2,ls='--')    
    ax.fill_between(dates,cumsum,color=color2,alpha=.2)
    
        
    ax.set_ylabel('Cumalitive moment \n (dyne*cm)')    

    ax.tick_params(axis='x', labelrotation = 90,size=10)        
    

    

    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getFeatures_Explore(catalog,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs,nfft,):
    """
    Read waveforms from H5, calculate features prior to SpecUFEx
    Return dataframe with features indexed by datetime

    Parameters
    ----------
    catalog : pandas Dataframe
        Needed for event IDs; need columns called 'ev_ID' and 'timestamp'
    fs : int
        Sampling rate.
    nfft : int
        padding for spectra.

    Returns
    -------
    df : pandas Dataframe
        Data frame of features by event ID.
    """



    columns=['ev_ID','log10RSAM','SpecCentr','log10P2P','log10Var','Kurt','DomFreq']
    df = pd.DataFrame(columns=columns)


    for i,evID in enumerate(catalog.event_ID):


        if i%200==0:
            print(i,'/',len(catalog))
            
        wf_filter = getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)

        # date = pd.to_datetime(catalog.timestamp.iloc[i])


        RSAM = np.log10(np.sum(np.abs(wf_filter)))

        sc = np.mean(librosa.feature.spectral_centroid(y=np.array(wf_filter), sr=fs))


        ### calculate dominant frequency
        f = np.fft.fft(wf_filter)
        f_real = np.real(f)
        mag_spec = plt.magnitude_spectrum(f_real,Fs=fs, scale='linear',pad_to=nfft)[0]
        freqs = plt.magnitude_spectrum(f_real,Fs=fs, scale='linear',pad_to=nfft)[1]
        dominant_freq = freqs[np.where(mag_spec == mag_spec.max())]
        plt.close()

        var = np.log10(np.var(wf_filter))
        p2p = np.log10(np.max(wf_filter) - np.min(wf_filter))
        kurt = kurtosis(wf_filter)

        df = df.append(
                  {'ev_ID':evID,
                   # 'datetime_index':date,
                   'log10RSAM':RSAM,
                   'SpecCentr':sc,
                   'log10P2P':p2p,
                   'log10Var':var,
                   'Kurt':kurt,
                   'DomFreq':dominant_freq[0]},
                   ignore_index=True)

    # df = df.set_index('datetime_index')

    return df




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def calcSilhScore(cat00,range_n_clusters,numPCA,distMeasure = "SilhScore",Xtype='fingerprints',stand=True):
    """


    Parameters
    ----------

    range_n_clusters : range type - 2 : Kmax clusters
    numPCA : number of principal components to perform clustering on (if not on FPs)
    Xtype : cluster directly on fingerprints or components of PCA. The default is 'fingerprints'.


    Returns
    -------
    Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters.

    """

## Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters
## Returns altered cat00 dataframe with cluster labels and SS scores,
## Returns NEW catall dataframe with highest SS scores



## alt. X = 'PCA'

    if Xtype == 'fingerprints':
        X = linearizeFP(path_proj,outfile_name,cat00)
        pca_df = cat00
    elif Xtype == 'PCA':
        __, pca_df, X = PCAonFP(path_proj,outfile_name,cat00,numPCA=numPCA,stand=stand);

    maxSilScore = 0

    sse = []
    avgSils = []
    centers = []

    for n_clusters in range_n_clusters:

        print(f"kmeans on {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters,
                           max_iter = 500,
                           init='k-means++', #how to choose init. centroid
                           n_init=10, #number of Kmeans runs
                           random_state=0) #set rand state

        #get cluster labels
        cluster_labels_0 = kmeans.fit_predict(X)

        #increment labels by one to match John's old kmeans code
        cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

        #get euclid dist to centroid for each point
        sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
        sum_sqr_dist = sqr_dist.sum(axis=1)
        euc_dist = np.sqrt(sum_sqr_dist)

        #save centroids
        centers.append(kmeans.cluster_centers_ )

        #kmeans loss function
        sse.append(kmeans.inertia_)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

#         %  Silhouette avg
        avgSil = np.mean(sample_silhouette_values)

        # avgSil = np.median(sample_silhouette_values)

        avgSils.append(avgSil)
        if avgSil > maxSilScore:
            Kopt = n_clusters
            maxSilScore = avgSil
            cluster_labels_best = cluster_labels
            euc_dist_best = euc_dist
            ss_best       = sample_silhouette_values


    print(f"Best cluster: {Kopt}")
    pca_df['Cluster'] = cluster_labels_best
    pca_df['SS'] = ss_best
    pca_df['euc_dist'] = euc_dist_best


    ## make df for  top SS score rep events
    catall = getTopFCat(pca_df,topF=1,startInd=0,distMeasure = distMeasure)







    return pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




def findEventPairHypDist(cat,indi,indj):
    """
    cat : Dataframe, catalog with "event_ID"
    indi, indj : index of catalog
    
    See last answer in: https://stackoverflow.com/questions/45618544/how-to-calculate-3d-distance-including-altitude-between-two-points-in-geodjang
    
    """
    
    lat1 = cat.iloc[indi].lat
    long1 = cat.iloc[indi].long
    depth1 = cat.iloc[indi].depth_km
    
    lat2 = cat.iloc[indj].lat
    long2 = cat.iloc[indj].long
    depth2 = cat.iloc[indj].depth_km
    
    ep_dist_km = haversine.haversine((lat1,long1),(lat2,long2))
    
    hyp_dist_km = np.sqrt(ep_dist_km**2 + (depth2-depth1)**2)
    
    return hyp_dist_km

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################





def findStationHypDist(cat,ind,station_coord):
    """
    cat : Dataframe, catalog with "event_ID"
    ind : index of catalog
    station_coord  : tuple (lat, long)
    
    See last answer in: https://stackoverflow.com/questions/45618544/how-to-calculate-3d-distance-including-altitude-between-two-points-in-geodjang
    
    """
    
    lat1 = station_coord[0] #station lat
    long1 = station_coord[1] #station lon
    depth1 = station_coord[2]#station "depth" = 0
    
    lat2 = cat.iloc[ind].lat
    long2 = cat.iloc[ind].long
    depth2 = cat.iloc[ind].depth_km
    
    ep_dist_km = haversine.haversine((lat1,long1),(lat2,long2))
    
    hyp_dist_km = np.sqrt(ep_dist_km**2 + (depth2-depth1)**2)
    
    return hyp_dist_km

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotMedianBins(X,Y,color='k',binSize = .05, ax = None):

    if ax is None:
        ax = plt.gca()

    bins = np.arange(min(X),max(X),binSize)


    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(X,Y,'median', bins=bins)
    bin_std1, __, __ = stats.binned_statistic(X,Y,'std', bins=bins)


    ax.scatter(bin_edges1[:-1]+binSize/2,bin_means1, marker='x',color=color)

    ax.vlines(bin_edges1[:-1]+binSize/2,
               bin_means1-bin_std1,
               bin_means1+bin_std1,
               color=color)
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotMeanBins(X,Y,color='k',binSize = .05, ax = None):

    if ax is None:
        ax = plt.gca()

    bins = np.arange(min(X),max(X),binSize)


    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(X,Y,'mean', bins=bins)
    bin_std1, __, __ = stats.binned_statistic(X,Y,'std', bins=bins)


    ax.scatter(bin_edges1[:-1]+binSize/2,bin_means1, marker='x',color=color)

    ax.vlines(bin_edges1[:-1]+binSize/2,
               bin_means1-bin_std1,
               bin_means1+bin_std1,
               color=color)    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




def calcCC_Element_FP(catRep,SpecUFEx_H5_path,indi,indj):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.
                    



    Returns np.array

    '''

    evIDA = catRep.event_ID.iloc[indi]
    evIDB = catRep.event_ID.iloc[indj]


    #linearize fingerprints
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        fpA = MLout['fingerprints'].get(evIDA)[:]                
        fpB = MLout['fingerprints'].get(evIDB)[:]

        fpA = fpA.reshape(1,len(fpA)**2)[:][0]     
        fpB = fpB.reshape(1,len(fpB)**2)[:][0]            


    cc = correlate(fpA, fpB, 0)
    lag, max_cc = xcorr_max(cc,abs_max=False)#Determines if the largest value of the correlation function is returned, independent of it being positive (correlation) or negative (anti-correlation). If False the maximum returned is positive only.

    return max_cc,lag


from scipy.spatial import distance

def calcCC_Element_FP2(catRep,SpecUFEx_H5_path,indi,indj):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.
                    



    Returns np.array

    '''

    evIDA = catRep.event_ID.iloc[indi]
    evIDB = catRep.event_ID.iloc[indj]


    #linearize fingerprints
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        fpA = MLout['fingerprints'].get(evIDA)[:]                
        fpB = MLout['fingerprints'].get(evIDB)[:]

        fpA = fpA.reshape(1,len(fpA)**2)[:][0]     
        fpB = fpB.reshape(1,len(fpB)**2)[:][0]            


    distance = distance.jensenshannon(fpA/max(fpA),fpB/max(fpB))

    return distance



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def calcCC_Element_WF(catRep,indi,indj,shift_cc,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs):
    '''
    catRep   : (pandas.Dataframe) catalog with event IDs

    shift_cc : (int) Number of samples to shift for cross correlation.
                    The cross-correlation will consist of 2*shift+1 or
                    2*shift samples. The sample with zero shift will be in the middle.
                    



    Returns np.array

    '''


    evIDA = catRep.event_ID.iloc[indi]
    evIDB = catRep.event_ID.iloc[indj]


    wf_A = getWF(evIDA,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)
    wf_B = getWF(evIDB,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)

    cc = correlate(wf_A, wf_B, shift_cc)
    lag, max_cc = xcorr_max(cc,abs_max=False)#Determines if the largest value of the correlation function is returned, independent of it being positive (correlation) or negative (anti-correlation). If False the maximum returned is positive only.

    return max_cc,lag

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotClusWF5(cat_clust,cat_clus_PM,lag_mat,lag_index,color,dataH5_path,station,channel,fmin,fmax,tmin,tmax,sampling_rate,alpha=1,fontsize=10,offset=True,ax=None):
    ## filtered, time-trimmed
    
    if ax is None:
        ax = plt.gca()
    
    clus_id_list = list(cat_clust.event_ID)
    
    
    
    for j, evID in enumerate(cat_clus_PM.event_ID):
        
        date = str(pd.Timestamp(cat_clus_PM.timestamp.iloc[j]).date())
        r =  cat_clus_PM.RID.iloc[j]
#         CCWF =  cat_clus_PM.medCCWF.iloc[j]        
#         CCFP =  cat_clus_PM.medCCFP.iloc[j]    
        
        
        
        wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,sampling_rate)
        end_time_s = len(wf)#//sampling_rate

        wf_norm = wf / np.max(np.abs(wf))
        wf_zeromean = wf_norm - np.mean(wf_norm)
        
        wf_lag = lagWF(wf_zeromean, lag_mat[:,j], index_wf=lag_index)

        
        ## No offset waveforms
        if offset:
            offset = len(cat_clust)/7
            wf_offset = (wf_lag) + j*offset#1.5

        else:
            offset = 0
            wf_offset = (wf_lag)

            
        ## new
        if r =='0' and evID in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='b') 
            if offset:
                ax.text(end_time_s,j*offset,s=f"{evID}-{r}",color='b',fontsize=14)            
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='b',fontsize=10)                
        
#         ##missing
#         elif r =='0' and evID not in clus_id_list:
#             ax.plot(wf_offset,lw=1,alpha=alpha,c='grey') 
#             if offset:
# #                 ax.text(end_time_s,j*offset,s=f"{evID} - {date}",color='grey',fontsize=14)            
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='gray',fontsize=10)                
        
        ## NOT recovered 
        elif evID not in clus_id_list and r!='0':
            ax.plot(wf_offset,lw=1,alpha=alpha,c='lightsalmon')              
            if offset:                            
                ax.text(end_time_s,j*offset,s=f"{evID}-{r}",color='lightsalmon',fontsize=14)            
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='lightsalmon',fontsize=10)                
                
        ## recovered 
        elif r !='0' and evID in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='red')              
            if offset:                            
                ax.text(end_time_s,j*offset,s=f"{evID}-{r}",color='red',fontsize=14)                
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='red',fontsize=10)                
                                        
                
        ## recovered 
        else:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='gray')              
            if offset:
                            
                ax.text(end_time_s,j*offset,s=f"{evID} ",color='gray',fontsize=14)                
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='gray',fontsize=10)                
                        
        
        
        ax.set_ylabel('Normalized amplitude')    
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks(list(range(0,end_time_s+1,sampling_rate)))
        ax.set_xticklabels(list(range(0,int(np.ceil(end_time_s/sampling_rate))+1)))               
        ax.set_xlabel(f'Time (s)')   
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotClusWF6(cat_clust,cat_clus_PM,lag_mat,lag_index,color,dataH5_path,station,channel,fmin,fmax,tmin,tmax,sampling_rate,alpha=1,fontsize=10,offset=True,ax=None):
    ## filtered, time-trimmed
    
    if ax is None:
        ax = plt.gca()
    
    clus_id_list = list(cat_clust.event_ID)
    
    
    
    for j, evID in enumerate(cat_clus_PM.event_ID):
        
        date = str(pd.Timestamp(cat_clus_PM.timestamp.iloc[j]).date())
        r =  cat_clus_PM.RID.iloc[j]
#         CCWF =  cat_clus_PM.medCCWF.iloc[j]        
#         CCFP =  cat_clus_PM.medCCFP.iloc[j]    
        
        
        
        wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,sampling_rate)
        end_time_s = len(wf)#//sampling_rate

        wf_norm = wf / np.max(np.abs(wf))
        wf_zeromean = wf_norm - np.mean(wf_norm)
        
        wf_lag = lagWF(wf_zeromean, lag_mat[:,j], index_wf=lag_index)

        
        ## No offset waveforms
        if offset:
            offset = len(cat_clust)/7
            wf_offset = (wf_lag) + j*offset#1.5

        else:
            offset = 0
            wf_offset = (wf_lag)

            
        ## new
        if r =='0' and evID in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='r') 
#             if offset:
#                 ax.text(end_time_s,j*offset,s=f"{date}",color='b',fontsize=14)                            
#                 ax.text(end_time_s,j*offset,s=f"{evID}-{r}",color='b',fontsize=14)            
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='b',fontsize=10)                
        
#         ##missing
#         elif r =='0' and evID not in clus_id_list:
#             ax.plot(wf_offset,lw=1,alpha=alpha,c='grey') 
#             if offset:
# #                 ax.text(end_time_s,j*offset,s=f"{evID} - {date}",color='grey',fontsize=14)            
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='gray',fontsize=10)                
        
        ## NOT recovered 
        elif evID not in clus_id_list and r!='0':
            ax.plot(wf_offset,lw=1,alpha=alpha,c='lightsalmon')              
#             if offset:                            
#                 ax.text(end_time_s,j*offset,s=f"{date}",color='lightsalmon',fontsize=14)                                            
#                 ax.text(end_time_s,j*offset,s=f"{evID}-{r}",color='lightsalmon',fontsize=14)            
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='lightsalmon',fontsize=10)                
                
        ## recovered 
        elif r !='0' and evID in clus_id_list:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='b')              
#             if offset:                 
#                 ax.text(end_time_s,j*offset,s=f"{date}",color='r',fontsize=14)                                            
#                 ax.text(end_time_s,j*offset,s=f"{evID}-{r}",color='red',fontsize=14)                
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='red',fontsize=10)                
                                        
                
        ## recovered 
        else:
            ax.plot(wf_offset,lw=1,alpha=alpha,c='steelblue')              
#             if offset:
                            
#                 ax.text(end_time_s,j*offset,s=f"{date}",color='gray',fontsize=14)                                                
#                 ax.text(end_time_s,j*offset,s=f"{evID} ",color='gray',fontsize=14)                
#                 ax.text(end_time_s,j*offset,s=f"{CCWF:.3f};{CCFP:.3f}",color='gray',fontsize=10)                
                        
        
        
        ax.set_ylabel('Normalized amplitude')    
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks(list(range(0,end_time_s+1,sampling_rate)))
        ax.set_xticklabels(list(range(0,int(np.ceil(end_time_s/sampling_rate))+1)))               
        ax.set_xlabel(f'Time (s)')   
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotCCDistancesfromMat(ccmat,cc_mat_FP,size=100, ax=None):
    
    
    if ax is None:
        ax = plt.gca()
        
        
    medianCCWF_list = []
    nrows = len(ccmat[:])
    for i, row in enumerate(ccmat[:]):
        row_2 = np.delete(row, i)    
        medianRow = np.median(row_2)
        medianCCWF_list.append(medianRow)



    medianCCFP_list = []
    nrows = len(cc_mat_FP[:]) 
    for i, row in enumerate(cc_mat_FP[:]):
        row_2 = np.delete(row, i)    
        medianRow = np.median(row_2)
        medianCCFP_list.append(medianRow)
    
        
        
    ax.scatter(medianCCWF_list,
                medianCCFP_list,
                edgecolor='gray',
                facecolor='None',               
                linestyle='None',
                s=size)
    


    point = (0,0)
    slope = 1
    ax.axline(xy1=point,slope=slope,color='k', linestyle='--')

#     ax.legend(markerscale=1)
    ax.axis('equal')
    ax.set_xlim(0,1.03)
    ax.set_ylim(0,1.03)
    
#     ax.set_ylim(np.min([0,min(cat_clus.medCCWF),min(cat_spec_trim_missing.medCCWF)]),1.03)  
    ax.set_ylabel('CC$_{median}$ fingerprint',labelpad=10)
    ax.set_xlabel('CC$_{median}$ waveform')    





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################





def plotScatterMap(cat, scale = .01,ax=None):

    if ax is None:
        ax = plt.gca()


    radii = [scale*calcRadius(m)**2 for m in cat.magnitude]
    cat.radius = radii



    cat_rep = cat[cat.RID!='0'].copy()
    cat_rep = cat_rep[cat_rep.RID!='9999'].copy()
    cat_new = cat[cat.RID=='0'].copy()
    cat_BG = cat[cat.RID=='9999'].copy()
    



    ## Split into magnitude bins - Background seismicity
    cat_0 = cat_BG[cat_BG.magnitude<1]

    cat_1 = cat_BG[cat_BG.magnitude>=1]
    cat_1 = cat_1[cat_1.magnitude<2]

    cat_2 = cat_BG[cat_BG.magnitude>=2]
    cat_2 = cat_2[cat_2.magnitude<3]

    cat_3 = cat_BG[cat_BG.magnitude>=3]


    ## Split into magnitude bins - RES seismicity
    cat_rep_0 = cat_rep[cat_rep.magnitude<1]

    cat_rep_1 = cat_rep[cat_rep.magnitude>=1]
    cat_rep_1 = cat_rep_1[cat_rep_1.magnitude<2]

    cat_rep_2 = cat_rep[cat_rep.magnitude>=2]
    cat_rep_2 = cat_rep_2[cat_rep_2.magnitude<3]

    cat_rep_3 = cat_rep[cat_rep.magnitude>=3]
    
    
    ## Split into magnitude bins - NEW seismicity
    cat_new_0 = cat_new[cat_new.magnitude<1]

    cat_new_1 = cat_new[cat_new.magnitude>=1]
    cat_new_1 = cat_new_1[cat_new_1.magnitude<2]

    cat_new_2 = cat_new[cat_new.magnitude>=2]
    cat_new_2 = cat_new_2[cat_new_2.magnitude<3]

    cat_new_3 = cat_new[cat_new.magnitude>=3]    




    #Background
    s0=1
    s1=10
    s2=100
    s3=500


    ax.scatter(cat_0.long, cat_0.lat,edgecolor='gray',facecolor='None',s=s0,label='Mw0')
    ax.scatter(cat_1.long, cat_1.lat,edgecolor='gray',facecolor='None',s=s1,label='Mw1')
    ax.scatter(cat_2.long, cat_2.lat,edgecolor='gray',facecolor='None',s=s2,label='Mw2')
    ax.scatter(cat_3.long, cat_3.lat,edgecolor='gray',facecolor='None',s=s3,label='Mw3')
    
    ax.scatter(cat_rep_0.long, cat_rep_0.lat,edgecolor='k',facecolor='None',s=s0)
    ax.scatter(cat_rep_1.long, cat_rep_1.lat,edgecolor='k',facecolor='None',s=s1)
    ax.scatter(cat_rep_2.long, cat_rep_2.lat,edgecolor='k',facecolor='None',s=s2)
    ax.scatter(cat_rep_3.long, cat_rep_3.lat,edgecolor='k',facecolor='None',s=s3)
    
    
    ax.scatter(cat_new_0.long, cat_new_0.lat,edgecolor='r',facecolor='None',s=s0)
    ax.scatter(cat_new_1.long, cat_new_1.lat,edgecolor='r',facecolor='None',s=s1)
    ax.scatter(cat_new_2.long, cat_new_2.lat,edgecolor='r',facecolor='None',s=s2)
    ax.scatter(cat_new_3.long, cat_new_3.lat,edgecolor='r',facecolor='None',s=s3)  
    

    
  

    ax.grid('on')



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




def plotScatterMapDepth(cat, scale = .01,ax=None):

    if ax is None:
        ax = plt.gca()


    radii = [scale*calcRadius(m)**2 for m in cat.magnitude]
    cat['radius'] = radii


    #Get already cataloged REQS
    cat_rep = cat[cat.RID!='0']
    cat_rep = cat_rep[cat_rep.RID!='9999']
    cat_new = cat[cat.RID=='0']
    cat_BG = cat[cat.RID=='9999']
    
    

    ## Split into magnitude bins - Background seismicity
    cat_0 = cat_BG[cat_BG.magnitude<1]

    cat_1 = cat_BG[cat_BG.magnitude>=1]
    cat_1 = cat_1[cat_1.magnitude<2]

    cat_2 = cat_BG[cat_BG.magnitude>=2]
    cat_2 = cat_2[cat_2.magnitude<3]

    cat_3 = cat_BG[cat_BG.magnitude>=3]


    ## Split into magnitude bins - RES seismicity
    cat_rep_0 = cat_rep[cat_rep.magnitude<1]

    cat_rep_1 = cat_rep[cat_rep.magnitude>=1]
    cat_rep_1 = cat_rep_1[cat_rep_1.magnitude<2]

    cat_rep_2 = cat_rep[cat_rep.magnitude>=2]
    cat_rep_2 = cat_rep_2[cat_rep_2.magnitude<3]

    cat_rep_3 = cat_rep[cat_rep.magnitude>=3]



    ## Split into magnitude bins - RES seismicity
    cat_new_0 = cat_new[cat_new.magnitude<1]

    cat_new_1 = cat_new[cat_new.magnitude>=1]
    cat_new_1 = cat_new_1[cat_new_1.magnitude<2]

    cat_new_2 = cat_new[cat_new.magnitude>=2]
    cat_new_2 = cat_new_2[cat_new_2.magnitude<3]

    cat_new_3 = cat_new[cat_new.magnitude>=3]
    #Background
    s0=1
    s1=10
    s2=100
    s3=500


    ax.scatter(cat_0.long, cat_0.depth_km,edgecolor='gray',facecolor='None',s=s0,label='Mw0')
    ax.scatter(cat_1.long, cat_1.depth_km,edgecolor='gray',facecolor='None',s=s1,label='Mw1')
    ax.scatter(cat_2.long, cat_2.depth_km,edgecolor='gray',facecolor='None',s=s2,label='Mw2')
    ax.scatter(cat_3.long, cat_3.depth_km,edgecolor='gray',facecolor='None',s=s3,label='Mw3')
    
    ax.scatter(cat_rep_0.long, cat_rep_0.depth_km,edgecolor='k',facecolor='None',s=s0)
    ax.scatter(cat_rep_1.long, cat_rep_1.depth_km,edgecolor='k',facecolor='None',s=s1)
    ax.scatter(cat_rep_2.long, cat_rep_2.depth_km,edgecolor='k',facecolor='None',s=s2)
    ax.scatter(cat_rep_3.long, cat_rep_3.depth_km,edgecolor='k',facecolor='None',s=s3)
    
    
    ax.scatter(cat_new_0.long, cat_new_0.depth_km,edgecolor='r',facecolor='None',s=s0)
    ax.scatter(cat_new_1.long, cat_new_1.depth_km,edgecolor='r',facecolor='None',s=s1)
    ax.scatter(cat_new_2.long, cat_new_2.depth_km,edgecolor='r',facecolor='None',s=s2)
    ax.scatter(cat_new_3.long, cat_new_3.depth_km,edgecolor='r',facecolor='None',s=s3)   
    

 

    ax.grid('on')
    ax.invert_yaxis()


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




def calcDfStats(cat,timeMin=3/12,NCl = 2):
    '''
    Inputs:
    cat : dataframe
    timeMin : minimum recurrence interval (years) - remove bursts
    NCl : Min number of events in a sequence
    
    '''

    cat_stats = pd.DataFrame()



    for i, clus in enumerate(np.unique(cat.Cluster_2)):

        if i%50==0:
            print(f"{i}/{len(np.unique(cat.Cluster_2))}")

        df_cl = cat[cat.Cluster_2==clus]

        ######################################################################################################
        ###############################   Difference from mean magnitude of criteria cluster   ###############
        ######################################################################################################
        mag_avg                        = np.mean(df_cl.magnitude)
        df_cl['mag_diff']              = [mag - mag_avg for mag in df_cl.magnitude]

        df_cl['mag_avg']              = [mag_avg for mag in df_cl.magnitude]

        df_cl['moment']               = [calcMom(dmag) for dmag in df_cl.magnitude]

        df_cl['moment_avg']            = [np.mean(df_cl.moment) for dmag in df_cl.moment] 


        S=0
        S_temp_list = []
        for Mo in df_cl.moment:
            S_temp = 10**-2.46 * Mo ** .17
            S_temp_list.append(S_temp)
            S += S_temp
        S = S / len(df_cl)



        df_cl['slip_avg']               = [S for s in df_cl.moment]  
        df_cl['slip']                   = S_temp_list         


        ######################################################################################################
        ###############################   How close to centroid of criteria cluster  #########################
        ######################################################################################################


        ## How close to centroid
        ## Location Centroid
        lon_ar = np.array(df_cl.long)
        lat_ar = np.array(df_cl.lat)
        centroid = (np.median(lon_ar),np.median(lat_ar))

        #
        ### DISTANCE to CENTROID
        df_cl['epi_dist_m'] = [latlon2meter((la,lo),(centroid[1],centroid[0])) for lo, la in zip(lon_ar,lat_ar)]## args in lat lon

        #
        ### Depth Difference  
        df_cl['dist_m'] = [np.sqrt((1000*(d - np.mean(df_cl.depth_km)))**2+ll_m**2) for d,ll_m in zip(df_cl.depth_km,df_cl['epi_dist_m'])]   




        ##### calc interevent time #####  
        df_cl['datetime'] = [pd.Timestamp(t) for t in df_cl.timestamp];
        df_cl.sort_values(by='datetime',inplace=True)
        time_diff_yr = [int(d)*1e-9/ 3600 / 24 / 365.25 for d in np.diff(df_cl['datetime'])] #converting ns to years
        time_diff_yr.insert(0,0) 
        df_cl['time_diff_yr'] = time_diff_yr


        df_cl2 = df_cl.iloc[1:][df_cl.iloc[1:].time_diff_yr > timeMin]
        df_cl2 = df_cl2.append(df_cl.iloc[0])
        df_cl2.sort_index(inplace=True)

        time_std = np.std(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_std_yr'] = time_std

        time_var = np.var(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_var_yr'] = time_var
        
        time_med = np.median(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_median_yr'] = time_med        

        time_mean = np.mean(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_mean_yr'] = time_mean   
        if time_mean > 0:

            df_cl2['CVr'] = time_std /   time_mean   

            SR = S / time_mean

            df_cl2['slipRate']            = [SR for S in df_cl2.time_mean_yr] 


            if len(df_cl2) >= NCl:

                cat_stats = cat_stats.append(df_cl2)


#                 cat_stats = cat_stats.sort_values(['CVr','RID'],ascending=True)


    cat_stats['numEvents'] = [len(cat_stats[cat_stats.Cluster_2==clus]) for clus in cat_stats.Cluster_2]
    
    print(len(cat),len(cat_stats))
    return cat_stats



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def calcDfStats2(cat,timeMin=3/12,NCl = 2):
    '''
    Inputs:
    cat : dataframe
    timeMin : minimum recurrence interval (years) - remove bursts
    NCl : Min number of events in a sequence
    
    '''

    cat_stats = pd.DataFrame()



    for i, clus in enumerate(np.unique(cat.RID)):

        if i%50==0:
            print(f"{i}/{len(np.unique(cat.RID))}")

        df_cl = cat[cat.RID==clus]

        ######################################################################################################
        ###############################   Difference from mean magnitude of criteria cluster   ###############
        ######################################################################################################
        mag_avg                        = np.mean(df_cl.magnitude)
        df_cl['mag_diff']              = [mag - mag_avg for mag in df_cl.magnitude]

        df_cl['mag_avg']              = [mag_avg for mag in df_cl.magnitude]

        df_cl['moment']               = [calcMom(dmag) for dmag in df_cl.magnitude]

        df_cl['moment_avg']            = [np.mean(df_cl.moment) for dmag in df_cl.moment] 

        df_cl['ruptureRadius']            = [np.mean(df_cl.moment) for dmag in df_cl.moment] 

        S=0
        S_temp_list = []
        for Mo in df_cl.moment:
            S_temp = 10**-2.46 * Mo ** .17
            S_temp_list.append(S_temp)
            S += S_temp
        S = S / len(df_cl)



        df_cl['slip_avg']               = [S for s in df_cl.moment]  
        df_cl['slip']                   = S_temp_list     


        ######################################################################################################
        ###############################   How close to centroid of criteria cluster  #########################
        ######################################################################################################


        ## How close to centroid
        ## Location Centroid
        lon_ar = np.array(df_cl.long)
        lat_ar = np.array(df_cl.lat)
        centroid = (np.median(lon_ar),np.median(lat_ar))

        #
        ### DISTANCE to CENTROID
        df_cl['epi_dist_m'] = [latlon2meter((la,lo),(centroid[1],centroid[0])) for lo, la in zip(lon_ar,lat_ar)]## args in lat lon

        #
        ### Depth Difference  
        df_cl['dist_m'] = [np.sqrt((1000*(d - np.mean(df_cl.depth_km)))**2+ll_m**2) for d,ll_m in zip(df_cl.depth_km,df_cl['epi_dist_m'])]   




        ##### calc interevent time #####  
        df_cl['datetime'] = [pd.Timestamp(t) for t in df_cl.timestamp];
        df_cl.sort_values(by='datetime',inplace=True)
        time_diff_yr = [int(d)*1e-9/ 3600 / 24 / 365.25 for d in np.diff(df_cl['datetime'])] #converting ns to years
        time_diff_yr.insert(0,0) 
        df_cl['time_diff_yr'] = time_diff_yr


        df_cl2 = df_cl.iloc[1:][df_cl.iloc[1:].time_diff_yr > timeMin]
        df_cl2 = df_cl2.append(df_cl.iloc[0])
        df_cl2.sort_index(inplace=True)

        time_std = np.std(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_std_yr'] = time_std

        time_var = np.var(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_var_yr'] = time_var
        
        time_med = np.median(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_median_yr'] = time_med          

        time_mean = np.mean(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_mean_yr'] = time_mean   
        if time_mean > 0:

            df_cl2['CVr'] = time_std /   time_mean   

            SR = S / time_mean

            df_cl2['slipRate']            = [SR for S in df_cl2.time_mean_yr] 


            if len(df_cl2) >= NCl:

                cat_stats = cat_stats.append(df_cl2)


#                 cat_stats = cat_stats.sort_values(['CVr','RID'],ascending=True)


    cat_stats['numEvents'] = [len(cat_stats[cat_stats.RID==clus]) for clus in cat_stats.RID]
    
    print(len(cat),len(cat_stats))
    return cat_stats

######################################################################################################
######################################################################################################
######################################################################################################

def calcDfRID2Stats(cat,timeMin=3/12,NCl = 2):
    '''
    Inputs:
    cat : dataframe
    timeMin : minimum recurrence interval (years) - remove bursts
    NCl : Min number of events in a sequence
    
    '''

    cat_stats = pd.DataFrame()



    for i, clus in enumerate(np.unique(cat.RID2)):

        if i%50==0:
            print(f"{i}/{len(np.unique(cat.RID2))}")

        df_cl = cat[cat.RID2==clus].copy()

        ######################################################################################################
        ###############################   Difference from mean magnitude of criteria cluster   ###############
        ######################################################################################################
        mag_avg                        = np.mean(df_cl.magnitude)
        df_cl['mag_diff']              = [mag - mag_avg for mag in df_cl.magnitude]

        df_cl['mag_avg']              = [mag_avg for mag in df_cl.magnitude]

        df_cl['moment']               = [calcMom(dmag) for dmag in df_cl.magnitude]

        df_cl['moment_avg']            = [np.mean(df_cl.moment) for dmag in df_cl.moment] 


        S=0
        S_temp_list = []
        for Mo in df_cl.moment:
            S_temp = 10**-2.46 * Mo ** .17
            S_temp_list.append(S_temp)
            S += S_temp
        S = S / len(df_cl)



        df_cl['slip_avg']               = [S for s in df_cl.moment]  
        df_cl['slip']                   = S_temp_list         


        ######################################################################################################
        ###############################   How close to centroid of criteria cluster  #########################
        ######################################################################################################


        ## How close to centroid
        ## Location Centroid
        lon_ar = np.array(df_cl.long)
        lat_ar = np.array(df_cl.lat)
        centroid = (np.median(lon_ar),np.median(lat_ar))

        #
        ### DISTANCE to CENTROID
        df_cl['epi_dist_m'] = [latlon2meter((la,lo),(centroid[1],centroid[0])) for lo, la in zip(lon_ar,lat_ar)]## args in lat lon

        #
        ### Depth Difference  
        df_cl['dist_m'] = [np.sqrt((1000*(d - np.mean(df_cl.depth_km)))**2+ll_m**2) for d,ll_m in zip(df_cl.depth_km,df_cl['epi_dist_m'])]   




        ##### calc interevent time #####  
        df_cl['datetime'] = [pd.Timestamp(t) for t in df_cl.timestamp];
        df_cl.sort_values(by='datetime',inplace=True)
        time_diff_yr = [int(d)*1e-9/ 3600 / 24 / 365.25 for d in np.diff(df_cl['datetime'])] #converting ns to years
        time_diff_yr.insert(0,0) 
        df_cl['time_diff_yr'] = time_diff_yr


        df_cl2 = df_cl.iloc[1:][df_cl.iloc[1:].time_diff_yr > timeMin]
        df_cl2 = df_cl2.append(df_cl.iloc[0])
        df_cl2.sort_index(inplace=True)

        time_std = np.std(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_std_yr'] = time_std

        time_var = np.var(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_var_yr'] = time_var
        
        time_med = np.median(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_median_yr'] = time_med        

        time_mean = np.mean(df_cl2.time_diff_yr.iloc[1:])
        df_cl2['time_mean_yr'] = time_mean   
        if time_mean > 0:

            df_cl2['CVr'] = time_std /   time_mean   

            SR = S / time_mean

            df_cl2['slipRate']            = [SR for S in df_cl2.time_mean_yr] 


            if len(df_cl2) >= NCl:

                cat_stats = cat_stats.append(df_cl2)


#                 cat_stats = cat_stats.sort_values(['CVr','RID'],ascending=True)


    cat_stats['numEvents'] = [len(cat_stats[cat_stats.RID2==clus]) for clus in cat_stats.RID2]
    
    print(len(cat),len(cat_stats))
    return cat_stats



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def rotateMapView(cat,angle,lat0,lon0):

    lat = cat.lat
    lon = cat.long

    dlat = lat - lat0
    dlon = lon - lon0

    y= dlat*111.1949;
    x= dlon*np.cos(lat*np.pi/180)*111.1949;

    # dlat and dlon being the difference between lat/lon origin and earthquake epicenter. 
    # lat is the latitude of the earthquake location (since longitude is latitude dependent). 

    #Then rotate, with PHI being the angle you like to rotate about (clockwise from north):
    phiA= (angle-90)*np.pi/180;

    # then plot distance along fault vs. depth: 
    xx= x * np.cos(phiA)- y * np.sin(phiA)

#     cat['dist_along_strike_km'] = xx
    # plt.scatter(xx,cat.depth_km)
    
    return xx


def plotScatterMapDepthAlongStrike(cat,lat0,lon0,angle = 48, scale = .01,ax=None):

    if ax is None:
        ax = plt.gca()

    cat['dist_along_strike_km'] = rotateMapView(cat,angle,lat0,lon0)

        
    radii = [scale*calcRadius(m)**2 for m in cat.magnitude]
    cat['radius'] = radii
    


    #Get already cataloged REQS
    cat_rep = cat[cat.RID!='0'].copy()
    cat_rep = cat_rep[cat_rep.RID!='9999'].copy()
    cat_new = cat[cat.RID=='0'].copy()
    cat_BG = cat[cat.RID=='9999'].copy()
    
    

    ## Split into magnitude bins - Background seismicity
    cat_0 = cat_BG[cat_BG.magnitude<1]

    cat_1 = cat_BG[cat_BG.magnitude>=1]
    cat_1 = cat_1[cat_1.magnitude<2]

    cat_2 = cat_BG[cat_BG.magnitude>=2]
    cat_2 = cat_2[cat_2.magnitude<3]

    cat_3 = cat_BG[cat_BG.magnitude>=3]


    ## Split into magnitude bins - RES seismicity
    cat_rep_0 = cat_rep[cat_rep.magnitude<1]

    cat_rep_1 = cat_rep[cat_rep.magnitude>=1]
    cat_rep_1 = cat_rep_1[cat_rep_1.magnitude<2]

    cat_rep_2 = cat_rep[cat_rep.magnitude>=2]
    cat_rep_2 = cat_rep_2[cat_rep_2.magnitude<3]

    cat_rep_3 = cat_rep[cat_rep.magnitude>=3]



    ## Split into magnitude bins - RES seismicity
    cat_new_0 = cat_new[cat_new.magnitude<1]

    cat_new_1 = cat_new[cat_new.magnitude>=1]
    cat_new_1 = cat_new_1[cat_new_1.magnitude<2]

    cat_new_2 = cat_new[cat_new.magnitude>=2]
    cat_new_2 = cat_new_2[cat_new_2.magnitude<3]

    cat_new_3 = cat_new[cat_new.magnitude>=3]
    #Background
    s0=1
    s1=10
    s2=100
    s3=500


    ax.scatter(cat_0.dist_along_strike_km, cat_0.depth_km,edgecolor='gray',facecolor='None',s=s0,label='Mw0')
    ax.scatter(cat_1.dist_along_strike_km, cat_1.depth_km,edgecolor='gray',facecolor='None',s=s1,label='Mw1')
    ax.scatter(cat_2.dist_along_strike_km, cat_2.depth_km,edgecolor='gray',facecolor='None',s=s2,label='Mw2')
    ax.scatter(cat_3.dist_along_strike_km, cat_3.depth_km,edgecolor='gray',facecolor='None',s=s3,label='Mw3')
    
    ax.scatter(cat_rep_0.dist_along_strike_km, cat_rep_0.depth_km,edgecolor='k',facecolor='None',s=s0)
    ax.scatter(cat_rep_1.dist_along_strike_km, cat_rep_1.depth_km,edgecolor='k',facecolor='None',s=s1)
    ax.scatter(cat_rep_2.dist_along_strike_km, cat_rep_2.depth_km,edgecolor='k',facecolor='None',s=s2)
    ax.scatter(cat_rep_3.dist_along_strike_km, cat_rep_3.depth_km,edgecolor='k',facecolor='None',s=s3)
    
    
    ax.scatter(cat_new_0.dist_along_strike_km, cat_new_0.depth_km,edgecolor='r',facecolor='None',s=s0)
    ax.scatter(cat_new_1.dist_along_strike_km, cat_new_1.depth_km,edgecolor='r',facecolor='None',s=s1)
    ax.scatter(cat_new_2.dist_along_strike_km, cat_new_2.depth_km,edgecolor='r',facecolor='None',s=s2)
    ax.scatter(cat_new_3.dist_along_strike_km, cat_new_3.depth_km,edgecolor='r',facecolor='None',s=s3)   
    

 

    ax.grid('on')
    ax.invert_yaxis()
    
    return cat

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################





##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


# import faults



def plotInsetMapCA(ax=None):
    
    if ax is None:
        ax= plt.gca(projection=ccrs.PlateCarree())
        
        
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    df = gpd.read_file('/Users/theresasawi/Documents/11_Manuscripts/SA_REQS/data/raw/SanAndreas.kml', driver='KML');
    df2 = gpd.read_file('/Users/theresasawi/Documents/11_Manuscripts/SA_REQS/data/raw/SanAndreasBayAreasFaults.kml', driver='KML');


    cat_lon = (-121.218446, -120.82917399999999)
    cat_lat = (36.268261000000003, 36.651126999999995)

    ax.coastlines(linewidth=2)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)

    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '110m',
        edgecolor='black', facecolor='none',linewidth=2))

    df2.iloc[1:].plot(ax=ax,color='brown')
    df.iloc[1:].plot(ax=ax,color='brown')


    ax.vlines([cat_lon[0], cat_lon[1]], cat_lat[0],cat_lat[1], linestyles='solid', colors='red')
    ax.hlines([cat_lat[0], cat_lat[1]], cat_lon[0],cat_lon[1], linestyles='solid', colors='red')
    ax.text(-121,36,'SAF',rotation=310,color='brown',fontsize=28)
    ax.text(-123,39.5,'CA',rotation=0,color='k',fontsize=28)
    ax.text(-118,39.5,'NV',rotation=0,color='k',fontsize=28)


    [i.set_linewidth(2) for i in ax.spines.values()];

#

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def gradeLocRID(cat,plot=0, verbose=1):
    grade_list = []
    grade_list_allevents = []

    for clus in np.unique(cat.RID):
        df_cl = cat[cat.RID==clus]

        ###find centroid
        df_cl_centroid = [np.median(df_cl.lat),np.median(df_cl.long)]
        df_cl_med_depth = np.median(df_cl.depth_km)

        ##get radius
        mag_max = np.max(df_cl.magnitude)
        r_m_A = calcRadius(mag_max) 
        r_m_B = calcRadius(mag_max) * 2


        ##find if other centroids in circle
        dist_epi_m   = [haversine.haversine(df_cl_centroid,[llat,llong],unit='m') for llat,llong in zip(df_cl.lat,df_cl.long)]
        dist_depth_m = [1000*(df_cl_med_depth - z) for z in df_cl.depth_km]
        dist_m = [np.sqrt(d**2 + z**2) for d,z in zip(dist_epi_m, dist_depth_m)]

        if np.all(dist_m < r_m_A):
            grade = 'A'
            grade_list.append('A')
            [grade_list_allevents.append('A') for a in range(len(df_cl))]

        elif np.all(dist_m < r_m_B):
            grade = 'B'  
            grade_list.append('B')
            [grade_list_allevents.append('B') for a in range(len(df_cl))]


        else:
            grade = 'C' 
            grade_list.append('C')
            [grade_list_allevents.append('C') for a in range(len(df_cl))]

        if verbose:
            print(f"Cluster {clus}; Grade {grade}")



        if plot:

            phi=np.arange(0,6.28,.001)
            ## x/y radii
            AA = (r_m_A)*np.cos(phi)+df_cl_centroid[0]
            BA = (r_m_A)*np.sin(phi)+df_cl_centroid[1]

            AB = (r_m_B)*np.cos(phi)+df_cl_centroid[0]
            BB = (r_m_B)*np.sin(phi)+df_cl_centroid[1]

            ## depth radii
            AA2 = (r_m_A)*np.cos(phi)+df_cl_centroid[0]
            BA2 = (r_m_A)*np.sin(phi)+df_cl_med_depth

            AB2 = (r_m_B)*np.cos(phi)+df_cl_centroid[0]
            BB2 = (r_m_B)*np.sin(phi)+df_cl_med_depth


            fig, axes = plt.subplots(ncols=2,figsize=(8,6))
            plt.subplots_adjust(wspace=.5)

            axes[0].axis('equal')
            axes[1].axis('equal')    

        #     Plot centroid
            axes[0].scatter(df_cl_centroid[0],df_cl_centroid[1],s=100)
            axes[1].scatter(df_cl_centroid[0],df_cl_med_depth,s=100)

            ## plot cluster
            plotLocZoom5(df_cl,df_cl,fontsize=16,ax=axes[0])     
            ## plot cluster boundary 
            axes[0].plot(AA, BA,'--',color='green')        
            axes[0].plot(AB, BB,'--',color='orange')    

            ## plot cluster
            plotDepthZoom8(df_cl,df_cl,fontsize=16,ax=axes[1])     
            ## plot cluster boundary 
            axes[1].plot(AA2, BA2,'--',color='green')        
            axes[1].plot(AB2, BB2,'--',color='orange')    
            axes[0].set_title(f"Grade:{grade}")


    
    cat['Grade'] = grade_list_allevents
    
    return cat
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    
def gradeLocCluster(cat,plot=0, verbose=1):

    grade_list = []
    grade_list_allevents = []

    for clus in np.unique(cat.Cluster_2):
        df_cl = cat[cat.Cluster_2==clus]

        ###find centroid
        df_cl_centroid = [np.median(df_cl.lat),np.median(df_cl.long)]
        df_cl_med_depth = np.median(df_cl.depth_km)

        ##get radius
        mag_max = np.max(df_cl.magnitude)
        r_m_A = calcRadius(mag_max) 
        r_m_B = calcRadius(mag_max) * 2


        ##find if other centroids in circle
        dist_epi_m   = [haversine.haversine(df_cl_centroid,[llat,llong],unit='m') for llat,llong in zip(df_cl.lat,df_cl.long)]
        dist_depth_m = [1000*(df_cl_med_depth - z) for z in df_cl.depth_km]
        dist_m = [np.sqrt(d**2 + z**2) for d,z in zip(dist_epi_m, dist_depth_m)]

        if np.all(dist_m < r_m_A):
            grade = 'A'
            grade_list.append('A')
            [grade_list_allevents.append('A') for a in range(len(df_cl))]

        elif np.all(dist_m < r_m_B):
            grade = 'B'  
            grade_list.append('B')
            [grade_list_allevents.append('B') for a in range(len(df_cl))]


        else:
            grade = 'C' 
            grade_list.append('C')
            [grade_list_allevents.append('C') for a in range(len(df_cl))]

        if verbose:
            print(f"Cluster {clus}; Grade {grade}")



        if plot:

            phi=np.arange(0,6.28,.001)
            ## x/y radii
            AA = (r_m_A)*np.cos(phi)+df_cl_centroid[0]
            BA = (r_m_A)*np.sin(phi)+df_cl_centroid[1]

            AB = (r_m_B)*np.cos(phi)+df_cl_centroid[0]
            BB = (r_m_B)*np.sin(phi)+df_cl_centroid[1]

            ## depth radii
            AA2 = (r_m_A)*np.cos(phi)+df_cl_centroid[0]
            BA2 = (r_m_A)*np.sin(phi)+df_cl_med_depth

            AB2 = (r_m_B)*np.cos(phi)+df_cl_centroid[0]
            BB2 = (r_m_B)*np.sin(phi)+df_cl_med_depth


            fig, axes = plt.subplots(ncols=2,figsize=(8,6))
            plt.subplots_adjust(wspace=.5)

            axes[0].axis('equal')
            axes[1].axis('equal')    

        #     Plot centroid
            axes[0].scatter(df_cl_centroid[0],df_cl_centroid[1],s=100)
            axes[1].scatter(df_cl_centroid[0],df_cl_med_depth,s=100)

            ## plot cluster
            plotLocZoom5(df_cl,df_cl,fontsize=16,ax=axes[0])     
            ## plot cluster boundary 
            axes[0].plot(AA, BA,'--',color='green')        
            axes[0].plot(AB, BB,'--',color='orange')    

            ## plot cluster
            plotDepthZoom8(df_cl,df_cl,fontsize=16,ax=axes[1])     
            ## plot cluster boundary 
            axes[1].plot(AA2, BA2,'--',color='green')        
            axes[1].plot(AB2, BB2,'--',color='orange')    
            axes[0].set_title(f"Grade:{grade}")


    
    cat['Grade'] = grade_list_allevents
    
    return cat



catRep_all = pd.read_csv('../data/catalogs/NCA_REPQcat_20210919_noMeta_v2.csv',header=0)


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def formatWS2021(pathCat):
    """
    This function reads earthquake data from a csv file located at pathCat and formats it into a new dataframe with additional columns for use in subsequent analyses.

    Parameters:
    pathCat (str): The path to the csv file containing earthquake data.

    Returns:
    pd.DataFrame
    """
    
    catRep_all = pd.read_csv(pathCat)    

    catRep_all['REQS_ID'] = catRep_all['DMAG']

    ## Get IDS from full REQS cat

    s = 0
    REQS_IDs = []
    REQS_evIDs = []
    isAnyRepeater_list = []

    allRep_evID = []

    for i, row in enumerate(catRep_all.YR): #YR column has # which indicates new cluster sequence

        if i==0:
            RID = catRep_all.DMAG.iloc[i]

        if '#' in row:
            RID = catRep_all.DMAG.iloc[i]

            s += 1
            #save Rpeeater Sequence ID
#             if i%500==0:
#                 print(RID)
        else:  
            ev  = catRep_all.iloc[i].evID #save event ID

            allRep_evID.append(str(int(ev)))
            REQS_evIDs.append(RID)
            isAnyRepeater_list.append(1)


    # from paper : " 27,675 repeating earthquakes grouped in 7,713 sequences"


    cat_rep_2014_df0 = pd.DataFrame({'RID':REQS_evIDs,
                                  'index':allRep_evID,
                                  'event_ID':allRep_evID,
                                    'isAnyRepeater':isAnyRepeater_list})

    cat_rep_2014_df0 = cat_rep_2014_df0.set_index('index')


    cat_rep_2014_df0['event_ID'] = [str(evv) for evv in cat_rep_2014_df0.event_ID]


    catRep_all['year'] = catRep_all['YR']
    catRep_all['month'] = catRep_all['MO']
    catRep_all['day'] = catRep_all['DY']
    catRep_all['hour'] = catRep_all['HR']
    catRep_all['minute'] = catRep_all['MN']
    catRep_all['second'] = catRep_all['SC']
    catRep_all['event_ID'] = [str(ev)[:-2] for ev in catRep_all['evID']]
    catRep_all['lat'] = catRep_all['LAT']
    catRep_all['long'] = catRep_all['LON']
    catRep_all['depth_km'] = catRep_all['DEP']
    catRep_all['magnitude'] = catRep_all['MAG']
    catRep_all['dmag'] = catRep_all['DMAG']
    catRep_all['dmage'] = catRep_all['DMAGE']
    catRep_all['dX'] = catRep_all['EX']
    catRep_all['dY'] = catRep_all['EY']
    catRep_all['dZ'] = catRep_all['EZ']
    
    # reorganize catalog so no more rows with #RID
    cat_rep_2014_df = cat_rep_2014_df0.merge(catRep_all,on='event_ID',how='left')

    cat_rep_2014_df['timestamp'] = pd.to_datetime(cat_rep_2014_df[['year','month','day','hour','minute','second']])

    cat_rep_2014_df['timestamp_index'] = pd.to_datetime(cat_rep_2014_df[['year','month','day','hour','minute','second']])
    cat_rep_2014_df = cat_rep_2014_df.set_index('timestamp_index')

    cat_rep_2014_df.sort_index(inplace=True)



    print(len(cat_rep_2014_df), 'repeater earthquakes in', s+1, ' sequences')

    return cat_rep_2014_df


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def removeEndedRES2(cat, endYear = 2015):
    

    # for year in range(1985,2021):
    for year in [endYear]:
        cat_reloc_graded_stats_cont = pd.DataFrame()

        for cl in np.unique(cat.RID):
            df_cl = cat[cat.RID==cl]

            lendf_cl = len(df_cl)

            time_mean = df_cl.time_mean_yr.iloc[0] *3

            time_end = df_cl.index[-1] 

            if (time_end.date()+pd.Timedelta(time_mean*365.25,'day')).year >= year:



                cat_reloc_graded_stats_cont = cat_reloc_graded_stats_cont.append(df_cl)

        
        print(len(cat),len(cat_reloc_graded_stats_cont))
        print(len(cat)-len(cat_reloc_graded_stats_cont), ' events removed')        
    
    return cat_reloc_graded_stats_cont

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def removeEndedRES(cat, endYear = 2015):
    
    for year in [endYear]:
        cat_reloc_graded_stats_cont = pd.DataFrame()

        for cl in np.unique(cat.Cluster_2):
            df_cl = cat[cat.Cluster_2==cl]

            lendf_cl = len(df_cl)

            time_mean = df_cl.time_mean_yr.iloc[0] *3

            time_end = df_cl.index[-1] 

            if (time_end.date()+pd.Timedelta(time_mean*365.25,'day')).year >= year:



                cat_reloc_graded_stats_cont = cat_reloc_graded_stats_cont.append(df_cl)

        print(len(cat),len(cat_reloc_graded_stats_cont))
        print(len(cat)-len(cat_reloc_graded_stats_cont), ' events removed')        

    return cat_reloc_graded_stats_cont

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def getProportionWS21(cat,cat_WS21):
    
    lenWS = len(cat_WS21)
    
    lenCatInWS = len(cat[cat.event_ID.isin(cat_WS21.event_ID)])
    
    return lenCatInWS / lenWS
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def getNumberInWS21(cat,cat_WS21):
    
    lenCatInWS = len(cat[cat.event_ID.isin(cat_WS21.event_ID)])
    
    return lenCatInWS 

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def ClusterExpand2(catAllClusAll, fp_df,lenData,dataH5_path, station,channel,fmin,fmax,tmin,tmax,sampling_rate,verbose=0):
    '''
    Inputs
    
    catAllClusAll : catalog of all criteria clusters
    fp_df : catalog of all events in study region    
    
    
    '''
    cat_discard_all = pd.DataFrame()

    clus_list_all = np.unique(catAllClusAll.Cluster)        

    for i, clus in enumerate(clus_list_all): 
    # for i, clus in enumerate(r_clus):

        #these are the clustered events
        cat_clus = catAllClusAll[catAllClusAll.Cluster==clus]
        cat_clus.sort_index(inplace=True)
        cluster = cat_clus.Cluster.iloc[0]

        ##get template event to compare new events to -- it's just the most recent of the largest ones
        max_mag = np.max(cat_clus.magnitude)
        cat_template = cat_clus[cat_clus.magnitude==max_mag].tail(1) #DF with single template event
    ######################################################################################################
    ######################################################################################################
        # Get missing events
        # Get missing events
        # Get missing events
        # Get missing events
        ##These are the unclustered events in a rectangle marked by clustered events    
        mag_avg = np.mean(cat_clus.magnitude)  
        r_m = calcRadius(mag_avg)
        # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
        r = r_m / 111139 

        maxlat  = max(cat_clus.lat)+r
        maxlong = max(cat_clus.long)+r
        minlat  = min(cat_clus.lat)-r
        minlong = min(cat_clus.long)-r
        maxDepth = max(cat_clus.depth_km)+r_m/1000
        minDepth = min(cat_clus.depth_km)-r_m/1000   

        ## get local earthquakes based on radius-magnitude
        cat_spec_trim = fp_df[fp_df.long<=maxlong]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.lat<=maxlat]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.long>=minlong]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.lat>=minlat] 
        cat_spec_trim = cat_spec_trim[cat_spec_trim.depth_km<=maxDepth]    
        cat_spec_trim = cat_spec_trim[cat_spec_trim.depth_km>minDepth]                


        missing_IDs = [x for x in list(cat_spec_trim.event_ID) if x not in list(catAllClusAll.event_ID)]

        cat_spec_trim_missing = cat_spec_trim[cat_spec_trim.event_ID.isin(missing_IDs)]
        cat_spec_trim_missing.sort_index(inplace=True)


    ######################################################################################################
    ################################      Get CC scores for original cluster    ##########################
    ######################################################################################################



        #df to keep added events
        cat_keep = cat_clus
        if i==0:
            cat_keep_all = cat_keep

        #df of events that were not added
        cat_discard = pd.DataFrame()



        ##find CC of clusteded events 
        ccmat, lag_mat = calcCCMatrix(
            cat_clus,
            lenData,
            dataH5_path,
            station,
            channel,
            fmin,
            fmax,
            tmin,tmax,
            sampling_rate)  


        #get median of row, minus the 1 on the diagonal
        medianCCWF_list = []
        nrows = len(ccmat[:])
        for i, row in enumerate(ccmat[:]):
            row_2 = np.delete(row, i)    
            medianRow = np.median(row_2)
            medianCCWF_list.append(medianRow)

        ccwf_med_orig = np.median(medianCCWF_list)
        if verbose:
            print('Cluster',cluster,' \n Orig CCmed', ccwf_med_orig)



        #now go through each missing event and see if the CC is still above CC Threshold
        for i, evM in enumerate(cat_spec_trim_missing.event_ID):
#             print(i)
    #         if evM not in list(cat_all.event_ID) and evM not in list(cat_clus_ALL_stats.event_ID):
            if evM not in list(catAllClusAll.event_ID):

                missing_event_cat = cat_spec_trim_missing[cat_spec_trim_missing.event_ID==evM]
                magnitude_eM = float(missing_event_cat.magnitude)

                cat_pm = cat_template.append(missing_event_cat)


                ccmat, lag_matpm = calcCCMatrix(
                    cat_pm,
                    lenData,
                    dataH5_path,
                    station,
                    channel,
                    fmin,
                    fmax,
                    tmin,tmax,
                    sampling_rate)  


                medianCCWF_list = []
                nrows = len(ccmat[:])
                for i, row in enumerate(ccmat[:]):
                    row_2 = np.delete(row, i)    
                    medianRow = np.median(row_2)
                    medianCCWF_list.append(medianRow)


                ## If adding the new waveform improves the CC or is above .9, then keep it, else discard
    #             print('\n CCmed =', np.median(medianCCWF_list), 'when adding event', evM)
                if  np.median(medianCCWF_list)>ccwf_med_orig or np.median(medianCCWF_list)>.9:


                    cat_keep_mag_max = np.max(cat_keep.magnitude)        
    #                     cat_keep_mag_med = np.median(cat_keep.magnitude)
    #                     cat_keep_mag_std = np.std(cat_keep.magnitude)  

                    mag_thresh = cat_keep_mag_max - .33
                    mag_thresh2 = cat_keep_mag_max + .33


                    ##and if the magnitude of the event is >= the other events
                    if magnitude_eM>=mag_thresh and magnitude_eM<=mag_thresh2:
                    ##Keep event! make sure its cluster is the same as the orig clustered ones
                        missing_event_cat.Cluster = cluster
                        if verbose:
                            print('add event!')
                        cat_keep = cat_keep.append(missing_event_cat)

                    else:
                        if verbose:

                            print('Event', evM, 'below magnitude threshold')
                        cat_discard = cat_discard.append(missing_event_cat)

                else:
                    if verbose:
                    
                        print('Event', evM, 'below CC threshold')
                    #### check magnitudes of kept 

            else:
                if verbose:
                    print('Event', evM, 'in catalog already')

        ## Keep all events        
    #     cat_all = cat_clus.append(cat_spec_trim_missing)
    #     cat_all.sort_index(inplace=True)
        if verbose:
            print("\n \t",len(cat_keep)-len(cat_clus), "events added")
        cat_keep_all = cat_keep_all.append(cat_keep)
        cat_discard_all = cat_discard_all.append(cat_discard)

    print('done')
    return cat_keep_all

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def makeWfTemplate(cat_clus,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs=100):
    """
    Get largest magnitude event from cluster and get its WF
    """
    
    magMax = np.max(cat_clus.magnitude)
    
    evID = cat_clus[cat_clus.magnitude==magMax].event_ID.iloc[0]
    
    wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)

    
    return wf
    
    
    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def RemoveMagnitudes(catalog, verbose = 1):

    cat_final = pd.DataFrame()


    for i, clus in enumerate(np.unique(catalog.Cluster)):
        if verbose:
        
            if i%30==0:
                print(i,'/',len(np.unique(catalog.Cluster)))
        df_cl = catalog[catalog.Cluster==clus]

        # remove += 0.3 median 
        medMag = np.median(df_cl.magnitude)
        df_cl_temp = df_cl[df_cl.magnitude <= medMag + .3]
        df_cl_temp = df_cl[df_cl.magnitude >= medMag - .3]   

        if len(df_cl_temp)>=2:
            cat_final=cat_final.append(df_cl_temp)

    if verbose:
        print(len(cat_final),len(catalog))
    return cat_final

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def meters2deg(d_m,lat):
    R=6378137
    d_d = d_m/(R*np.cos(np.pi*lat/180))
    return d_d * 180 / np.pi    


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def ClusterExpand3(catAllClusAll, fp_df,lenData,dataH5_path, station,channel,fmin,fmax,tmin,tmax,sampling_rate,verbose=0):
    '''
    Inputs

    catAllClusAll : catalog of all criteria clusters
    fp_df : catalog of all events in study region    


    '''
    cat_discard_all = pd.DataFrame()

    clus_list_all = np.unique(catAllClusAll.Cluster)        

    for i, clus in enumerate(clus_list_all): 
    # for i, clus in enumerate(r_clus):

        #these are the clustered events
        cat_clus = catAllClusAll[catAllClusAll.Cluster==clus]
        cat_clus.sort_index(inplace=True)
        cluster = cat_clus.Cluster.iloc[0]

        ##get template event to compare new events to -- it's just the most recent of the largest ones
        max_mag = np.max(cat_clus.magnitude)
        cat_template = cat_clus[cat_clus.magnitude==max_mag].tail(1) #DF with single template event
    ######################################################################################################
    ######################################################################################################
        # Get missing events
        # Get missing events
        # Get missing events
        # Get missing events
        ##These are the unclustered events in a rectangle marked by clustered events    
        mag_avg = np.median(cat_clus.magnitude)  
        r_m = calcRadius(mag_avg)
        r =  meters2deg(r_m,np.median(cat_clus.lat))

        maxlat  = max(cat_clus.lat)+r
        maxlong = max(cat_clus.long)+r
        minlat  = min(cat_clus.lat)-r
        minlong = min(cat_clus.long)-r
        maxDepth = max(cat_clus.depth_km)+r_m/1000
        minDepth = min(cat_clus.depth_km)-r_m/1000   

        ## get local earthquakes based on radius-magnitude
        cat_spec_trim = fp_df[fp_df.long<=maxlong]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.lat<=maxlat]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.long>=minlong]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.lat>=minlat] 
        cat_spec_trim = cat_spec_trim[cat_spec_trim.depth_km<=maxDepth]    
        cat_spec_trim = cat_spec_trim[cat_spec_trim.depth_km>minDepth]                


        missing_IDs = [x for x in list(cat_spec_trim.event_ID) if x not in list(catAllClusAll.event_ID)]

        cat_spec_trim_missing = cat_spec_trim[cat_spec_trim.event_ID.isin(missing_IDs)]
        cat_spec_trim_missing.sort_index(inplace=True)


    ######################################################################################################
    ################################      Get CC scores for original cluster    ##########################
    ######################################################################################################



        #df to keep added events
        cat_keep = cat_clus
        if i==0:
            cat_keep_all = cat_keep

        #df of events that were not added
        cat_discard = pd.DataFrame()



        ##find CC of clusteded events 
        ccmat, lag_mat = calcCCMatrix(
            cat_clus,
            lenData,
            dataH5_path,
            station,
            channel,
            fmin,
            fmax,
            tmin,tmax,
            sampling_rate)  


        #get median of row, minus the 1 on the diagonal
        medianCCWF_list = []
        nrows = len(ccmat[:])
        for i, row in enumerate(ccmat[:]):
            row_2 = np.delete(row, i)    
            medianRow = np.median(row_2)
            medianCCWF_list.append(medianRow)

        ccwf_med_orig = np.median(medianCCWF_list)
        if verbose:
            print('Cluster',cluster,' \n Orig CCmed', ccwf_med_orig)



        #now go through each missing event and see if the CC is still above CC Threshold
        for i, evM in enumerate(cat_spec_trim_missing.event_ID):
    #             print(i)
    #         if evM not in list(cat_all.event_ID) and evM not in list(cat_clus_ALL_stats.event_ID):
            if evM not in list(catAllClusAll.event_ID):

                missing_event_cat = cat_spec_trim_missing[cat_spec_trim_missing.event_ID==evM]
                magnitude_eM = float(missing_event_cat.magnitude)

                cat_pm = cat_template.append(missing_event_cat)


                ccmat, lag_matpm = calcCCMatrix(
                    cat_pm,
                    lenData,
                    dataH5_path,
                    station,
                    channel,
                    fmin,
                    fmax,
                    tmin,tmax,
                    sampling_rate)  


                medianCCWF_list = []
                nrows = len(ccmat[:])
                for i, row in enumerate(ccmat[:]):
                    row_2 = np.delete(row, i)    
                    medianRow = np.median(row_2)
                    medianCCWF_list.append(medianRow)


                ## If adding the new waveform improves the CC or is above .9, then keep it, else discard
    #             print('\n CCmed =', np.median(medianCCWF_list), 'when adding event', evM)
                if  np.median(medianCCWF_list)>ccwf_med_orig or np.median(medianCCWF_list)>.9:


                    cat_keep_mag_max = np.max(cat_keep.magnitude)        
    #                     cat_keep_mag_med = np.median(cat_keep.magnitude)
    #                     cat_keep_mag_std = np.std(cat_keep.magnitude)  

                    mag_thresh = cat_keep_mag_max - .3
                    mag_thresh2 = cat_keep_mag_max + .3


                    ##and if the magnitude of the event is >= the other events
                    if magnitude_eM>=mag_thresh and magnitude_eM<=mag_thresh2:
                    ##Keep event! make sure its cluster is the same as the orig clustered ones
                        missing_event_cat.Cluster = cluster
                        if verbose:
                            print('add event!')
                        cat_keep = cat_keep.append(missing_event_cat)

                    else:
                        if verbose:

                            print('Event', evM, 'below magnitude threshold')
                        cat_discard = cat_discard.append(missing_event_cat)

                else:
                    if verbose:

                        print('Event', evM, 'below CC threshold')
                    #### check magnitudes of kept 

            else:
                if verbose:
                    print('Event', evM, 'in catalog already')

        ## Keep all events        
    #     cat_all = cat_clus.append(cat_spec_trim_missing)
    #     cat_all.sort_index(inplace=True)
        if verbose:
            print("\n \t",len(cat_keep)-len(cat_clus), "events added")
        cat_keep_all = cat_keep_all.append(cat_keep)
        cat_discard_all = cat_discard_all.append(cat_discard)

    print('done')
    return cat_keep_all



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def ClusterExpand4(catAllClusAll, fp_df,lenData,dataH5_path, station,channel,fmin,fmax,tmin,tmax,sampling_rate,verbose=0):
    '''
    Inputs

    catAllClusAll : catalog of all criteria clusters
    fp_df : catalog of all events in study region    


    '''
    cat_discard_all = pd.DataFrame()

    clus_list_all = np.unique(catAllClusAll.Cluster)        

    for i, clus in enumerate(clus_list_all): 
    # for i, clus in enumerate(r_clus):

        #these are the clustered events
        cat_clus = catAllClusAll[catAllClusAll.Cluster==clus]
        cat_clus.sort_index(inplace=True)
        cluster = cat_clus.Cluster.iloc[0]

        ##get template event to compare new events to -- it's just the most recent of the largest ones
        max_mag = np.max(cat_clus.magnitude)
        cat_template = cat_clus[cat_clus.magnitude==max_mag].tail(1) #DF with single template event
    ######################################################################################################
    ######################################################################################################
        # Get missing events
        # Get missing events
        # Get missing events
        # Get missing events
        ##These are the unclustered events in a rectangle marked by clustered events    
        mag_avg = np.median(cat_clus.magnitude)  
        r_m = calcRadius(mag_avg)
        r =  meters2deg(r_m,np.median(cat_clus.lat))

        maxlat  = max(cat_clus.lat)
        maxlong = max(cat_clus.long)
        minlat  = min(cat_clus.lat)
        minlong = min(cat_clus.long)
        maxDepth = max(cat_clus.depth_km)
        minDepth = min(cat_clus.depth_km)

        ## get local earthquakes based on radius-magnitude
        cat_spec_trim = fp_df[fp_df.long<=maxlong]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.lat<=maxlat]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.long>=minlong]
        cat_spec_trim = cat_spec_trim[cat_spec_trim.lat>=minlat] 
        cat_spec_trim = cat_spec_trim[cat_spec_trim.depth_km<=maxDepth]    
        cat_spec_trim = cat_spec_trim[cat_spec_trim.depth_km>minDepth]                


        missing_IDs = [x for x in list(cat_spec_trim.event_ID) if x not in list(catAllClusAll.event_ID)]

        cat_spec_trim_missing = cat_spec_trim[cat_spec_trim.event_ID.isin(missing_IDs)]
        cat_spec_trim_missing.sort_index(inplace=True)


    ######################################################################################################
    ################################      Get CC scores for original cluster    ##########################
    ######################################################################################################



        #df to keep added events
        cat_keep = cat_clus
        if i==0:
            cat_keep_all = cat_keep

        #df of events that were not added
        cat_discard = pd.DataFrame()



        ##find CC of clusteded events 
        ccmat, lag_mat = calcCCMatrix(
            cat_clus,
            lenData,
            dataH5_path,
            station,
            channel,
            fmin,
            fmax,
            tmin,tmax,
            sampling_rate)  


        #get median of row, minus the 1 on the diagonal
        medianCCWF_list = []
        nrows = len(ccmat[:])
        for i, row in enumerate(ccmat[:]):
            row_2 = np.delete(row, i)    
            medianRow = np.median(row_2)
            medianCCWF_list.append(medianRow)

        ccwf_med_orig = np.median(medianCCWF_list)
        if verbose:
            print('Cluster',cluster,' \n Orig CCmed', ccwf_med_orig)



        #now go through each missing event and see if the CC is still above CC Threshold
        for i, evM in enumerate(cat_spec_trim_missing.event_ID):
    #             print(i)
    #         if evM not in list(cat_all.event_ID) and evM not in list(cat_clus_ALL_stats.event_ID):
            if evM not in list(catAllClusAll.event_ID):

                missing_event_cat = cat_spec_trim_missing[cat_spec_trim_missing.event_ID==evM]
                magnitude_eM = float(missing_event_cat.magnitude)

                cat_pm = cat_template.append(missing_event_cat)


                ccmat, lag_matpm = calcCCMatrix(
                    cat_pm,
                    lenData,
                    dataH5_path,
                    station,
                    channel,
                    fmin,
                    fmax,
                    tmin,tmax,
                    sampling_rate)  


                medianCCWF_list = []
                nrows = len(ccmat[:])
                for i, row in enumerate(ccmat[:]):
                    row_2 = np.delete(row, i)    
                    medianRow = np.median(row_2)
                    medianCCWF_list.append(medianRow)


                ## If adding the new waveform improves the CC or is above .9, then keep it, else discard
    #             print('\n CCmed =', np.median(medianCCWF_list), 'when adding event', evM)
                if  np.median(medianCCWF_list)>ccwf_med_orig or np.median(medianCCWF_list)>.9:


                    cat_keep_mag_max = np.max(cat_keep.magnitude)        
    #                     cat_keep_mag_med = np.median(cat_keep.magnitude)
    #                     cat_keep_mag_std = np.std(cat_keep.magnitude)  

                    mag_thresh = cat_keep_mag_max - .3
                    mag_thresh2 = cat_keep_mag_max + .3


                    ##and if the magnitude of the event is >= the other events
                    if magnitude_eM>=mag_thresh and magnitude_eM<=mag_thresh2:
                    ##Keep event! make sure its cluster is the same as the orig clustered ones
                        missing_event_cat.Cluster = cluster
                        if verbose:
                            print('add event!')
                        cat_keep = cat_keep.append(missing_event_cat)

                    else:
                        if verbose:

                            print('Event', evM, 'below magnitude threshold')
                        cat_discard = cat_discard.append(missing_event_cat)

                else:
                    if verbose:

                        print('Event', evM, 'below CC threshold')
                    #### check magnitudes of kept 

            else:
                if verbose:
                    print('Event', evM, 'in catalog already')

        ## Keep all events        
    #     cat_all = cat_clus.append(cat_spec_trim_missing)
    #     cat_all.sort_index(inplace=True)
        if verbose:
            print("\n \t",len(cat_keep)-len(cat_clus), "events added")
        cat_keep_all = cat_keep_all.append(cat_keep)
        cat_discard_all = cat_discard_all.append(cat_discard)

    print('done')
    return cat_keep_all



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def plotCircleLatLonError(cat_clus,indd):
    
    med_lat = np.median(cat_clus.lat)
    mag = cat_clus.magnitude.iloc[indd]
    X = cat_clus.long.iloc[indd]
    Y = cat_clus.lat.iloc[indd]
    
    dXm = cat_clus.dX.iloc[indd]
    dYm = cat_clus.dY.iloc[indd]
    
    dX = meters2deg(dXm,med_lat)
    dY = meters2deg(dYm,med_lat)  
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r = meters2deg(calcRadius(mag),med_lat)

    
    
    return (r)*np.cos(phi)+X, (r)*np.sin(phi)+Y, X, Y, dX,dY,r
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotCircleDepthLatLonError(cat_clus,indd):
    
    
    mag = cat_clus.magnitude.iloc[indd]
    X = cat_clus.long.iloc[indd]
    Z = cat_clus.depth_km.iloc[indd] * 1000
    
    
    med_lat = np.median(cat_clus.lat)
    dX = meters2deg(cat_clus.dX.iloc[indd],med_lat)
    dZ = cat_clus.dZ.iloc[indd] 
    phi=np.arange(0,6.28,.001)
    
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r = meters2deg(calcRadius(mag),med_lat)
    r_m = calcRadius(mag)
        
    
    return (r)*np.cos(phi)+X, (r_m)*np.sin(phi)+Z, X, Z, dX,dZ,r



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def plotLocLatLonError(cat_clust,cat_full,fontsize=16,ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_aspect('equal')
    
    med_lat = np.median(cat_clust.lat)
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    
    
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###
    ### Plot Missing circles in gray###   
    ## Get area of hypocenters 
    mag_avg = np.mean(cat_clust.magnitude)  
    
    r_m = calcRadius(mag_avg)
    # each degree the radius line of the Earth sweeps out corresponds to 111,139 meters
    r = meters2deg(r_m,med_lat)

    
    maxlat  = max(cat_clust.lat)+r
    maxlong = max(cat_clust.long)+r
    minlat  = min(cat_clust.lat)-r
    minlong = min(cat_clust.long)-r

    cat_spec_clus_trim = cat_full[cat_full.long<=maxlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat<=maxlat]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.long>=minlong]
    cat_spec_clus_trim = cat_spec_clus_trim[cat_spec_clus_trim.lat>=minlat]
    

    for indd in range(len(cat_spec_clus_trim)):

        A2, B2, X, Y, dX,dY, r = plotCircleLatLonError(cat_spec_clus_trim,indd)

        ax.plot(A2,B2, c='grey',ls='-' )

        ax.set_xlabel('Long')
        ax.set_ylabel('Lat') 

        dX = meters2deg(max(10,dX),med_lat)
        dY = meters2deg(max(10,dY),med_lat)
        ##plot error pars
        ax.scatter(X,Y, c='gray',ls='None',marker='x' )    
        ax.vlines(X,Y-dY, Y+dY, linestyles='-', colors='gray')
        ax.hlines(Y,X-dX, X+dX, linestyles='-', colors='gray')


    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A2, B2, X, Y, dX,dY, r = plotCircleLatLonError(cat_clust,indd)

        ax.plot(A2,B2, c='b',ls='-' )

        ax.set_xlabel('Long')
        ax.set_ylabel('Lat') 

        dX = meters2deg(max(10,dX),med_lat)
        dY = meters2deg(max(10,dY),med_lat)
        ##plot error pars
        ax.scatter(X,Y, c='gray',ls='None',marker='x' )    
        ax.vlines(X,Y-dY, Y+dY, linestyles='-', colors='gray')
        ax.hlines(Y,X-dX, X+dX, linestyles='-', colors='gray')

        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A2, B2, X, Y, dX,dY, r = plotCircleLatLonError(cat_clust_orig,indd)

        ax.plot(A2,B2, c='r',ls='-' )

        ax.set_xlabel('Long')
        ax.set_ylabel('Lat') 

        dX = meters2deg(max(10,dX),med_lat)
        dY = meters2deg(max(10,dY),med_lat)
        ##plot error pars
        ax.scatter(X,Y, c='gray',ls='None',marker='x' )    
        ax.vlines(X,Y-dY, Y+dY, linestyles='-', colors='gray')
        ax.hlines(Y,X-dX, X+dX, linestyles='-', colors='gray')
        ax.tick_params(axis='x', labelrotation = 45,size=8)   

    ax.grid('on')
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
def plotDepthLatLonError(cat_clust,cat_spec_trim_missing,fontsize=16,ax=None):

    if ax is None:
        ax = plt.gca()
        ax.axis('equal')
       
    
    cat_clust_orig = cat_clust[cat_clust.RID!='0']
    med_lat = np.median(cat_clust.lat)
 

    for indd in range(len(cat_spec_trim_missing)):

        A2, B2, X, Z, dX,dZ, r_m =  plotCircleDepthLatLonError(cat_spec_trim_missing,indd)

        ax.plot(A2,B2, c='lightsalmon',ls='-' )
        
        dX = meters2deg(max(10,dX),med_lat)
        dZ = max(10,dZ)
        ###error bars    
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')
        
        
    #### Missing, w/ RID
    cat_spec_trim_missing_RID = cat_spec_trim_missing[cat_spec_trim_missing.RID!='0']
    
    for indd in range(len(cat_spec_trim_missing_RID)):

        A2, B2, X, Z, dX,dZ, r_m = plotCircleDepthLatLonError(cat_spec_trim_missing_RID,indd)

        ax.plot(A2,B2, c='lightsalmon',ls='-' )
        
        dX = meters2deg(max(10,dX),med_lat)
        dZ = max(10,dZ)
        ###error bars    
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')


    

    

    ### Plot New circles in Blue###
    ### Plot New circles in Blue###
    ### Plot New circles in Blue###    
    
    for indd in range(len(cat_clust)):

        A2, B2, X, Z, dX,dZ, r_m = plotCircleDepthLatLonError(cat_clust,indd)

        ax.plot(A2,B2, c='b',ls='-' )
        
        dX = meters2deg(max(10,dX),med_lat)
        dZ = max(10,dZ)
        ###error bars    
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')



        
    ### Plot Recovered circles in Red###
    ### Plot Recovered circles in Red###    
    ### Plot Recovered circles in Red###    
    for indd in range(len(cat_clust_orig)):

        A2, B2, X, Z, dX,dZ, r_m = plotCircleDepthLatLonError(cat_clust_orig,indd)

        ax.plot(A2,B2, c='r',ls='-' )

        ###error bars 
        dX = meters2deg(max(10,dX),med_lat)
        dZ = max(10,dZ)        
        ax.scatter(X,Z, c='gray',ls='None',marker='x' )        
        ax.vlines(X,Z-dZ, Z+dZ, linestyles='-', colors='grey')
        ax.hlines(Z,X-dX, X+dX, linestyles='-', colors='grey')


        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)') 
    

    
    
    ax.set_xlabel(f'Long',labelpad=20,fontsize=fontsize)
    ax.set_ylabel(f'Depth (m)',labelpad=5,fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation = 45,size=8)   
    


    ax.invert_yaxis()
    ax.grid('on')
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    



def GradeLocErrorLatLon(cat,verbose=1,plot=0,removeEvents=1):
    grade_list = []
    grade_list_allevents = []

    cat_reloc_graded = pd.DataFrame()

    for clus in np.unique(cat.Cluster_2):
        if verbose:    
            print(f"Cluster {clus}")    
        df_cl = cat[cat.Cluster_2==clus]

        df_cl['depth_m'] = [1000 * d for d in df_cl.depth_km]
        ###find centroid
        df_cl_centroid = [np.median(df_cl.lat),np.median(df_cl.long)]
        df_cl_med_depth = np.median(df_cl.depth_m)
        
        med_lat = np.median(df_cl.lat)

        ##get radius
        med_max = np.median(df_cl.magnitude + .1)
        r_m_A = calcRadius(med_max)/2
        r_m_B = calcRadius(med_max)
        r_A = meters2deg(r_m_A,med_lat)
        r_B = meters2deg(r_m_B,med_lat)
#         r_A = r_m_A/ 111139
#         r_B = r_m_B/ 111139 




        ##find if other centroids in circle


        ### adjust error to 10m if error < 10m 
        dX_list = [max(X,10) for X in df_cl.dX]
        dY_list = [max(Y,10) for Y in df_cl.dY]
        dZ_list = [max(Z,10) for Z in df_cl.dZ]

        df_cl['dLong_adj'] = [meters2deg(x,med_lat) for x in dX_list]
        df_cl['dLat_adj'] = [meters2deg(y,med_lat) for y in dY_list]
        df_cl['dDepth_m_adj'] = [z for z in dZ_list]


        ## Whichever is negative, add error
        ## Whichever is positive, subtract error
        negX = df_cl.long[df_cl.long<=df_cl_centroid[1]] + df_cl['dLong_adj'][df_cl.long<=df_cl_centroid[1]]
        posX = df_cl.long[df_cl.long>df_cl_centroid[1]] - df_cl['dLong_adj'][df_cl.long>df_cl_centroid[1]]
        df_cl['long_adj'] = negX.append(posX)

        negY = df_cl.lat[df_cl.lat<=df_cl_centroid[0]] + df_cl['dLat_adj'][df_cl.lat<=df_cl_centroid[0]]
        posY = df_cl.lat[df_cl.lat>df_cl_centroid[0]] - df_cl['dLat_adj'][df_cl.lat>df_cl_centroid[0]]
        df_cl['lat_adj'] = negY.append(posY)

        negZ = df_cl.depth_m[df_cl.depth_m<=df_cl_med_depth] + df_cl['dDepth_m_adj'][df_cl.depth_m<=df_cl_med_depth]
        posZ = df_cl.depth_m[df_cl.depth_m>df_cl_med_depth] - df_cl['dDepth_m_adj'][df_cl.depth_m>df_cl_med_depth]
        df_cl['depth_m_adj'] = negZ.append(posZ)


        dist_epi_m       = [haversine.haversine(df_cl_centroid,[llat,llong],unit='m') for llat,llong in zip(df_cl.lat,df_cl.long)]
        dist_depth_m     = [(df_cl_med_depth - z) for z in df_cl.depth_m]
        dist_m           = [np.sqrt(d**2 + z**2) for d,z in zip(dist_epi_m, dist_depth_m)]

        dist_epi_adj_m   = [haversine.haversine(df_cl_centroid,[llat,llong],unit='m') for llat,llong in zip(df_cl.lat_adj,df_cl.long_adj)]
        dist_depth_adj_m = [(df_cl_med_depth - z) for z in df_cl.depth_m_adj]    
        dist_adj_m       = [np.sqrt(d**2 + z**2) for d,z in zip(dist_epi_adj_m, dist_depth_adj_m)]

        df_cl['dist_m'] = dist_m
        df_cl['dist_adj_m'] = dist_adj_m


        if np.all(dist_m < r_m_A):
            if verbose:
                print('initial grade A')
            grade = 'A'    
        elif np.all(dist_m < r_m_B):
            if verbose:
                print('initial grade B')                
            if np.all(dist_adj_m < r_m_A):            
                grade = 'A'               
            else:
                grade = 'B'      
        else:
            if verbose:
                print('initial grade C') 

            if np.all(dist_adj_m < r_m_B):            
                grade = 'B'           
            else:
                grade = 'C' 
                cat_temp = pd.DataFrame()
                i=0
                for dist in df_cl.dist_adj_m:

                    if dist <= r_m_B:
                        cat_temp = cat_temp.append(df_cl.iloc[i])
                    else:

                        if verbose:
                            print('Outside event removed')
                    i+=1 
                if len(cat_temp)>1:
                    if verbose:
                        print('removing',len(df_cl)-len(cat_temp),'events') 
                    df_cl = cat_temp
                    if verbose:
                        print('recalculating centroid...')
                    df_cl_centroid = [np.median(df_cl.lat),np.median(df_cl.long)]
                    df_cl_med_depth = np.median(df_cl.depth_m)                    
                    dist_epi_m   = [haversine.haversine(df_cl_centroid,[llat,llong],unit='m') for llat,llong in zip(df_cl.lat,df_cl.long)]
                    dist_depth_m = [(df_cl_med_depth - z) for z in df_cl.depth_m]
                    dist_m = [np.sqrt(d**2 + z**2) for d,z in zip(dist_epi_m, dist_depth_m)]

                    dist_epi_adj_m   = [haversine.haversine(df_cl_centroid,[llat,llong],unit='m') for llat,llong in zip(df_cl.lat_adj,df_cl.long_adj)]
                    dist_depth_adj_m = [(df_cl_med_depth - z) for z in df_cl.depth_m_adj]    
                    dist_adj_m = [np.sqrt(d**2 + z**2) for d,z in zip(dist_epi_adj_m, dist_depth_adj_m)]

                    df_cl['dist_m'] = dist_m
                    df_cl['dist_adj_m'] = dist_adj_m
                    if np.all(dist_m < r_m_A):
                        grade = 'A'
                    elif np.all(dist_m < r_m_B) or np.all(dist_adj_m < r_m_B):
                        if np.all(dist_adj_m < r_m_A):
                            grade = 'A'
                        else:
                            grade = 'B'  
                    else:
                        grade = 'C' 




                else:
                    if verbose:
                        print('cluster too short')
        if verbose:    
            print(f"Cluster {clus}; Grade {grade}")

        grade_list.append(grade)
        [grade_list_allevents.append(grade) for a in range(len(df_cl))]          
        cat_reloc_graded = cat_reloc_graded.append(df_cl)



        if plot:

            phi=np.arange(0,6.28,.001)
            ## x/y radii
            AA = (r_A)*np.cos(phi)+df_cl_centroid[1]
            BA = (r_A)*np.sin(phi)+df_cl_centroid[0]

            AB = (r_B)*np.cos(phi)+df_cl_centroid[1]
            BB = (r_B)*np.sin(phi)+df_cl_centroid[0]

            ## depth radii
            AA2 = (r_A)*np.cos(phi)+df_cl_centroid[1]
            BA2 = (r_m_A)*np.sin(phi)+df_cl_med_depth

            AB2 = (r_B)*np.cos(phi)+df_cl_centroid[1]
            BB2 = (r_m_B)*np.sin(phi)+df_cl_med_depth


            fig, axes = plt.subplots(ncols=2,figsize=(6,4))
            plt.subplots_adjust(wspace=.5)

            axes[0].axis('equal')

        #     Plot centroid
            axes[0].scatter(df_cl_centroid[1],df_cl_centroid[0],s=100)
            axes[1].scatter(df_cl_centroid[1],df_cl_med_depth,s=100)

            ## plot cluster
            plotLocLatLonError(df_cl,df_cl,fontsize=16,ax=axes[0])     



            ## plot cluster boundary 
            axes[0].plot(AA, BA,'--',color='green')        
            axes[0].plot(AB, BB,'--',color='orange')    
            axes[0].scatter(df_cl.long_adj, df_cl.lat_adj,color='r',marker='x')    

            ## plot cluster
#             ax[1].set_aspect(1/111139)
            plotDepthLatLonError(df_cl,df_cl,fontsize=16,ax=axes[1]) 
#             axes[1].set_aspect('equal')
            axes[1].set_aspect(meters2deg(1,med_lat))


            ## plot cluster boundary 
            axes[1].plot(AA2, BA2,'--',color='green')        
            axes[1].plot(AB2, BB2,'--',color='orange')  
            axes[1].scatter(df_cl.long_adj, df_cl.depth_m_adj,color='r',marker='x')        

            axes[0].set_title(f"Grade:{grade}", )
            axes[1].set_title(f"Cl:{clus}")        

            axes[0].grid('on')
            axes[1].grid('on')        
            axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))        
            axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))        


    print(len(cat),len(cat_reloc_graded))
    cat_reloc_graded['Grade'] = grade_list_allevents
    return cat_reloc_graded




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    



def GradeLocErrorMeters(reloc_cat,verbose=0,plot=0,removeEvents = 1):
    
    grade_list = []
    grade_list_allevents = []

    cat_reloc_graded = pd.DataFrame()

    for clus in np.unique(reloc_cat.Cluster_2):
        if verbose:    
            print(f"Cluster {clus}")    
        df_cl = reloc_cat[reloc_cat.Cluster_2==clus]

        ###find centroid
        df_cl_centroid = [np.median(df_cl.X),np.median(df_cl.Y),np.median(df_cl.Z)]




        ##get radius
        mag_max = np.max(df_cl.magnitude)
        r_m_A = calcRadius(mag_max) 
        r_m_B = calcRadius(mag_max)*2




        ##adjust for errors

        ### adjust error to 10m if error < 10m 
        dX_list = [max(X,10) for X in df_cl.dX]
        dY_list = [max(Y,10) for Y in df_cl.dY]
        dZ_list = [max(Z,10) for Z in df_cl.dZ]

        df_cl['dX_adj'] = dX_list
        df_cl['dY_adj'] = dY_list
        df_cl['dZ_adj'] = dZ_list

        ## Whichever is negative, add error
        ## Whichever is positive, subtract error
        negX = df_cl.X[df_cl.X<=0] + df_cl['dX_adj'][df_cl.X<=0]
        posX = df_cl.X[df_cl.X>0] - df_cl['dX_adj'][df_cl.X>0]
        df_cl['X_adj'] = negX.append(posX)

        negY = df_cl.Y[df_cl.Y<=0] + df_cl['dY_adj'][df_cl.Y<=0]
        posY = df_cl.Y[df_cl.Y>0] - df_cl['dY_adj'][df_cl.Y>0]
        df_cl['Y_adj'] = negY.append(posY)

        midZ = np.median(df_cl.Z)
        negZ = df_cl.Z[df_cl.Z<=midZ] + df_cl['dZ_adj'][df_cl.Z<=midZ]
        posZ = df_cl.Z[df_cl.Z>midZ] - df_cl['dZ_adj'][df_cl.Z>midZ]
        df_cl['Z_adj'] = negZ.append(posZ)

        dist_m = [euclidean([x,y,z],df_cl_centroid) for x,y,z in zip(df_cl.X,df_cl.Y,df_cl.Z)]
        dist_adj_m = [euclidean([x,y,z],df_cl_centroid) for x,y,z in zip(df_cl.X_adj,df_cl.Y_adj,df_cl.Z_adj)]

        df_cl['dist_m'] = dist_m
        df_cl['dist_adj_m'] = dist_adj_m


        if np.all(dist_m < r_m_A):
            if verbose:
                print('initial grade A')
            grade = 'A'    
        elif np.all(dist_m < r_m_B):
            if verbose:
                print('initial grade B')                
            if np.all(dist_adj_m < r_m_A):            
                grade = 'A'               
            else:
                grade = 'B'      
        else:
            if verbose:
                print('initial grade C') 

            if np.all(dist_adj_m < r_m_B):            
                grade = 'B'           
            else:
                grade = 'C' 
                cat_temp = pd.DataFrame()
                i=0
                for dist in df_cl.dist_adj_m:

                    if dist <= r_m_B:
                        cat_temp = cat_temp.append(df_cl.iloc[i])
                    else:
                        if verbose:
                            print('Outside event removed')
                    i+=1 
                if len(cat_temp)>1:
                    if verbose:
                        print('removing',len(df_cl)-len(cat_temp),'events') 
                    df_cl = cat_temp
                    if verbose:
                        print('recalculating centroid...')
                    df_cl_centroid = [np.median(df_cl.X),np.median(df_cl.Y),np.median(df_cl.Z)]
                    dist_m = [euclidean([x,y,z],df_cl_centroid) for x,y,z in zip(df_cl.X,df_cl.Y,df_cl.Z)]
                    dist_adj_m = [euclidean([x,y,z],df_cl_centroid) for x,y,z in zip(df_cl.X_adj,df_cl.Y_adj,df_cl.Z_adj)]
                    df_cl['dist_m'] = dist_m
                    df_cl['dist_adj_m'] = dist_adj_m
                    if np.all(dist_m < r_m_A):
                        grade = 'A'
                    elif np.all(dist_m < r_m_B) or np.all(dist_adj_m < r_m_B):
                        if np.all(dist_adj_m < r_m_A):
                            grade = 'A'
                        else:
                            grade = 'B'  
                    else:
                        grade = 'C' 




                else:
                    if verbose:
                        print('cluster too short')
        if verbose:    
            print(f"Cluster {clus}; Grade {grade}")

        grade_list.append(grade)
        [grade_list_allevents.append(grade) for a in range(len(df_cl))]          
        cat_reloc_graded = cat_reloc_graded.append(df_cl)



        if plot:

            phi=np.arange(0,6.28,.001)
            ## x/y radii
            AA = (r_m_A)*np.cos(phi)+df_cl_centroid[0]
            BA = (r_m_A)*np.sin(phi)+df_cl_centroid[1]

            AB = (r_m_B)*np.cos(phi)+df_cl_centroid[0]
            BB = (r_m_B)*np.sin(phi)+df_cl_centroid[1]

            ## depth radii
            AA2 = (r_m_A)*np.cos(phi)+df_cl_centroid[0]
            BA2 = (r_m_A)*np.sin(phi)+df_cl_centroid[2]

            AB2 = (r_m_B)*np.cos(phi)+df_cl_centroid[0]
            BB2 = (r_m_B)*np.sin(phi)+df_cl_centroid[2]


            fig, axes = plt.subplots(ncols=2,figsize=(6,4))
            plt.subplots_adjust(wspace=.5)

            axes[0].axis('equal')
            axes[1].axis('equal')    

        #     Plot centroid
            axes[0].scatter(df_cl_centroid[0],df_cl_centroid[1],s=100)
            axes[1].scatter(df_cl_centroid[0],df_cl_centroid[2],s=100)

            ## plot cluster
            plotLocZoomMeters(df_cl,df_cl,fontsize=16,ax=axes[0])     



            ## plot cluster boundary 
            axes[0].plot(AA, BA,'--',color='green')        
            axes[0].plot(AB, BB,'--',color='orange')    
            axes[0].scatter(df_cl.X_adj, df_cl.Y_adj,color='r',marker='x')    

            ## plot cluster
            plotDepthZoomMeters(df_cl,df_cl,fontsize=16,ax=axes[1])     
            ## plot cluster boundary 
            axes[1].plot(AA2, BA2,'--',color='green')        
            axes[1].plot(AB2, BB2,'--',color='orange')  
            axes[1].scatter(df_cl.X_adj, df_cl.Z_adj,color='r',marker='x')        

            axes[0].set_title(f"Grade:{grade}")
            axes[1].set_title(f"Cl:{clus}")        

    print(len(reloc_cat),len(cat_reloc_graded))
    cat_reloc_graded['Grade'] = grade_list_allevents
    
    return cat_reloc_graded



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def makeWfTemplate(cat_clus,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs=100):
    """
    Get largest magnitude event from cluster and get its WF
    """
    
    magMax = np.max(cat_clus.magnitude)
    
    evID = cat_clus[cat_clus.magnitude==magMax].event_ID.iloc[0]
    
    wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)

    
    return wf
    
    
    

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def ClusterConverge2(cat_expanded,dataH5_path,station,channel,fmin,fmax,tmin,tmax,verbose=1):
    
    # Part 6. Converge clusters

    clus_list_all = np.unique(cat_expanded.Cluster)

    clus_converge = pd.DataFrame()

    for i, clus in enumerate(clus_list_all):

        if verbose:
            if i%10==0:
                print(i,'/',len(clus_list_all))

         ## get df of clusters   
        cat_clusA = cat_expanded[cat_expanded.Cluster==clus]

        ## Location Centroid
        lon_ar = np.array(cat_clusA.long)
        lat_ar = np.array(cat_clusA.lat)
        centroidA = (np.median(lat_ar),np.median(lon_ar))


        for j, clus in enumerate(clus_list_all):
            if i!= j:
                cat_clusB = cat_expanded[cat_expanded.Cluster==clus]

                ## Location Centroid
                lon_ar = np.array(cat_clusB.long)
                lat_ar = np.array(cat_clusB.lat)
                centroidB = (np.median(lat_ar),np.median(lon_ar))   

                centroid_dist_m = latlon2meter((centroidA[0],centroidA[1]),(centroidB[0],centroidB[1]))## args in lat lon

                med_mag = np.median([np.median(cat_clusA.magnitude),np.median(cat_clusB.magnitude)])

                dist_thresh = calcRadius(med_mag)

                if centroid_dist_m < dist_thresh: #m            

                    wf_A = makeWfTemplate(cat_clusA,dataH5_path,station,channel,fmin,fmax,tmin,tmax)
                    wf_B = makeWfTemplate(cat_clusB,dataH5_path,station,channel,fmin,fmax,tmin,tmax)

                    cc = correlate(wf_A, wf_B, 1000)
                    lag, max_cc = xcorr_max(cc)

                    if max_cc >= .95:
    #                     print('adding cluster', clus)
                        clusA = cat_clusA.Cluster.iloc[0]
                        cat_clusA = cat_clusA.append(cat_clusB)
                        cat_clusA.loc[:,'Cluster'] = clusA


                clus_converge = clus_converge.append(cat_clusA).drop_duplicates('event_ID')    

    print('done')
    return clus_converge


    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def ClusterConverge3(cat_expanded,dataH5_path,station,channel,fmin,fmax,tmin,tmax,verbose=1):
    
    # Part 6. Converge clusters

    clus_list_all = np.unique(cat_expanded.Cluster)

    clus_converge = pd.DataFrame()

    for i, clus in enumerate(clus_list_all):

        if verbose:
            if i%10==0:
                print(i,'/',len(clus_list_all))

         ## get df of clusters   
        cat_clusA = cat_expanded[cat_expanded.Cluster==clus]

        ## Location Centroid
        lon_ar = np.array(cat_clusA.long)
        lat_ar = np.array(cat_clusA.lat)
        centroidA = (np.median(lat_ar),np.median(lon_ar))


        for j, clus in enumerate(clus_list_all):
            if i!= j:
                cat_clusB = cat_expanded[cat_expanded.Cluster==clus]

                ## Location Centroid
                lon_ar = np.array(cat_clusB.long)
                lat_ar = np.array(cat_clusB.lat)
                centroidB = (np.median(lat_ar),np.median(lon_ar))   

                centroid_dist_m = latlon2meter((centroidA[0],centroidA[1]),(centroidB[0],centroidB[1]))## args in lat lon

                med_mag = np.median([np.median(cat_clusA.magnitude),np.median(cat_clusB.magnitude)])

                dist_thresh = calcRadius(med_mag)

                if centroid_dist_m < dist_thresh: #m            

                    wf_A = makeWfTemplate(cat_clusA,dataH5_path,station,channel,fmin,fmax,tmin,tmax)
                    wf_B = makeWfTemplate(cat_clusB,dataH5_path,station,channel,fmin,fmax,tmin,tmax)

                    cc = correlate(wf_A, wf_B, 1000)
                    lag, max_cc = xcorr_max(cc)

                    if max_cc >= .95:
    #                     print('adding cluster', clus)
                        clusA = cat_clusA.Cluster.iloc[0]
                        cat_clusA = cat_clusA.append(cat_clusB)
                        cat_clusA.loc[:,'Cluster'] = clusA


                clus_converge = clus_converge.append(cat_clusA).drop_duplicates('event_ID')    

    print('done')
    return clus_converge


    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    
    
def addRID(cat,cat_rep_2014_df):
    RID_list = []
    count= 0
    for i, row in cat.iterrows():

        evID = str(row.event_ID)    
        if evID in list(cat_rep_2014_df.event_ID):

            count+=1
            df_ev = cat_rep_2014_df[cat_rep_2014_df.event_ID==evID].copy()

            RID_list.append(df_ev.RID[0])    

        else:
            RID_list.append('0')    

    print(count)    


    cat['RID'] = RID_list
    
    return cat


 
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def getYearlyCount(catalog, binSize='Y'):

    yealy_count_catalog = catalog.RID.resample(binSize).count()
    index2 = [yealy_count_catalog.index[i] - pd.DateOffset(months=11,days=30) for i in range(len(yealy_count_catalog))]
    yealy_count_catalog = yealy_count_catalog.set_axis(index2)
    
    return yealy_count_catalog
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
   
def getYearlyMedianMag(catalog, binSize='Y'):

    yealy_median_mag_catalog = catalog.magnitude.resample(binSize).median()
    index2 = [yealy_median_mag_catalog.index[i] - pd.DateOffset(months=11,days=30) for i in range(len(yealy_median_mag_catalog))]
    yealy_median_mag_catalog = yealy_median_mag_catalog.set_axis(index2)
    
    return yealy_median_mag_catalog
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
   
def removeEndedRES3(cat, endYear = 2015):
    

    # for year in range(1985,2021):
    for year in [endYear]:
        cat_reloc_graded_stats_cont = pd.DataFrame()

        for cl in np.unique(cat.RID2):
            df_cl = cat[cat.RID2==cl]

            time_mean = df_cl.time_mean_yr.iloc[0] *3

            time_end = df_cl.index[-1] 

            if (time_end.date()+pd.Timedelta(time_mean*365.25,'day')).year >= year:



                cat_reloc_graded_stats_cont = cat_reloc_graded_stats_cont.append(df_cl)

        
        print(len(cat),len(cat_reloc_graded_stats_cont))
        print(len(cat)-len(cat_reloc_graded_stats_cont), ' events removed')        
    
    return cat_reloc_graded_stats_cont
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
   
def removeEndedLateRES(cat, endYear = 2015):
    
    startYear = 1984
    
    cat_reloc_graded_stats_cont = pd.DataFrame()

    for cl in np.unique(cat.RID2):
        df_cl = cat[cat.RID2==cl]

        time_mean = df_cl.time_mean_yr.iloc[0] *3

        time_end = df_cl.index[-1] 

        if ((time_end.date()+pd.Timedelta(time_mean*365.25,'day')).year >= endYear) and ((time_end.date()-pd.Timedelta(time_mean*365.25,'day')).year <= startYear):



            cat_reloc_graded_stats_cont = cat_reloc_graded_stats_cont.append(df_cl)

        
        print(len(cat),len(cat_reloc_graded_stats_cont))
        print(len(cat)-len(cat_reloc_graded_stats_cont), ' events removed')        
    
    return cat_reloc_graded_stats_cont    
    
    
##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
    
    
    

def getYearlyMedianSlipRate(catalog, binSize='Y'):

    yealy_median_catalog = catalog.slipRate.resample(binSize).median()
    index2 = [yealy_median_catalog.index[i] - pd.DateOffset(months=11,days=30) for i in range(len(yealy_median_catalog))]
    yealy_median_catalog = yealy_median_catalog.set_axis(index2)
    
    return yealy_median_catalog

