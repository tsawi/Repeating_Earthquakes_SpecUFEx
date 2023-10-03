## p2_cluster_functions.py

## For cluster processing 
## from p2_cluster_functions import makeWfTemplate, ClusterExpand, ClusterConverge

## For Hierarchical clustering
## from p2_cluster_functions import linearizeFP, 
## from sklearn.cluster import AgglomerativeClustering


import h5py
import pandas as pd
import numpy as np
from obspy import read
from haversine import haversine
from scipy.signal import butter, lfilter
from obspy.signal.cross_correlation import correlate, xcorr_max
import datetime





####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------

# def latlon2meter(point1,point2):
#    """
#     Calculate the distance in meters between two points on the Earth's surface, given their latitude and longitude.

#     Parameters:
#     -----------
#     point1 : tuple
#         A tuple of two floats representing the latitude and longitude of the first point in degrees.
#     point2 : tuple
#         A tuple of two floats representing the latitude and longitude of the second point in degrees.

#     Returns:
#     --------
#     distance : float
#         The distance between the two points in meters, calculated using the Haversine formula.
#     """
    
#     return haversine.haversine(point1, point2) * 1e3

####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------

def latlon2meter_vec(coords1, coords2):
    
    """
    Calculate the distance in meters between two pairs of latitude and longitude coordinates.

    Parameters:
    -----------
    coords1 : tuple
        A tuple containing the latitude and longitude of the first point in degrees.
    coords2 : tuple
        A tuple containing the latitude and longitude of the second point in degrees.

    Returns:
    --------
    distance : float
        The distance in meters between the two points.

    Notes:
    ------
    This function uses vectorized distance calculation using numpy broadcasting.

    """
    
    lat1, lon1 = np.radians(coords1)
    lat2, lon2 = np.radians(coords2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = 6371000 * c
    return distance

####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------


def ClusterExpand(catAllClusAll, fp_df,lenData,dataH5_path, station,channel,fmin,fmax,tmin,tmax,sampling_rate,verbose=0):
    '''
    Expands a catalog of clusters by adding nearby events that improve the CC between the new event and the cluster.

    Inputs:

        catAllClusAll (pd.Dataframe): a catalog of all the criteria clusters.
        fp_df (pd.Dataframe): a catalog of all the events in the study region.
        lenData (int) : the length of the waveform data in seconds.
        dataH5_path (str): the path of the HDF5 file containing the waveform data.
        station (str): the station name used for the data in the HDF5 file.
        channel (str): the channel used for the data in the HDF5 file.
        fmin, fmax (int, float): the minimum and maximum frequencies used for filtering the waveform data.
        tmin, tmax (int, float): the minimum and maximum times (in seconds) relative to the earthquake origin time, used for windowing the waveform data.
        sampling_rate (int, float): the sampling rate of the waveform data.
        verbose (optional): if set to 1, print additional information. Default is 1
    '''
    
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H%M%S")  
    
    with open('clusterExpand_'+date_string+'.txt', 'w') as f:        
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
                print('Cluster',cluster,' \n Orig CCmed', ccwf_med_orig,file=f)



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

                                print('Event', evM, 'below magnitude threshold',file=f)
                            cat_discard = cat_discard.append(missing_event_cat)

                    else:
                        if verbose:

                            print('Event', evM, 'below CC threshold',file=f)
                        #### check magnitudes of kept 

                else:
                    if verbose:
                        print('Event', evM, 'in catalog already',file=f)

            ## Keep all events        
        #     cat_all = cat_clus.append(cat_spec_trim_missing)
        #     cat_all.sort_index(inplace=True)
            if verbose:
                print("\n \t",len(cat_keep)-len(cat_clus), "events added",file=f)
            cat_keep_all = cat_keep_all.append(cat_keep)
            cat_discard_all = cat_discard_all.append(cat_discard)

        print('done')
    return cat_keep_all

##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


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


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


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



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


def makeWfTemplate(cat_clus,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs=100):
    """
    Create a waveform template from the largest magnitude event in a cluster.

    Args:
    - cat_clus: Pandas dataframe object containing events in a cluster
    - dataH5_path: Path to .h5 file containing seismic data
    - station: Station code
    - channel: Channel code
    - fmin: Minimum frequency in Hz
    - fmax: Maximum frequency in Hz
    - tmin: Start time of waveform in UTCDateTime format
    - tmax: End time of waveform in UTCDateTime format
    - fs: Sampling frequency (default is 100)

    Returns:
    - wf: numpy array containing waveform data for the largest magnitude event in the cluster
    """
    
    magMax = np.max(cat_clus.magnitude)
    
    evID = cat_clus[cat_clus.magnitude==magMax].event_ID.iloc[0]
    
    wf = getWF(evID,dataH5_path,station,channel,fmin,fmax,tmin,tmax,fs)

    
    return wf
    


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def ClusterConverge(cat_expanded,dataH5_path,station,channel,fmin,fmax,tmin,tmax,verbose=1):
    """
    Cluster convergence algorithm to merge similar clusters based on location, magnitude and waveform cross-correlation.
    
    Parameters:
        cat_expanded (pd.DataFrame): Catalog containing expanded events.
        dataH5_path (str): Path to HDF5 data file containing waveforms.
        station (str): Station code.
        channel (str): Channel code.
        fmin (float): Minimum frequency in Hz for bandpass filter.
        fmax (float): Maximum frequency in Hz for bandpass filter.
        tmin (float): Start time in seconds for waveform window.
        tmax (float): End time in seconds for waveform window.
        verbose (int, optional): Verbosity level. Default is 0.
    
    Returns:
        pd.DataFrame: Catalog containing converged clusters.
    """    
    

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H%M%S")  
    
    with open('clusterConvergence_'+date_string+'.txt', 'w') as f:    
        print('cluster convergence...',file=f)
        clus_list_all = np.unique(cat_expanded.Cluster)

        clus_converge = pd.DataFrame()

        for i, clusA in enumerate(clus_list_all):

            if verbose:
                if i%100==0:
                    print(i,'/',len(clus_list_all))

             ## get df of clusters   
            cat_clusA = cat_expanded[cat_expanded.Cluster==clusA]

            ## Location Centroid
            lon_ar = np.array(cat_clusA.long)
            lat_ar = np.array(cat_clusA.lat)
            centroidA = (np.median(lat_ar),np.median(lon_ar))

            magA = np.median(cat_clusA.magnitude)

            for j, clusB in enumerate(clus_list_all):
                if i!= j:


                    cat_clusB = cat_expanded[cat_expanded.Cluster==clusB]


                    magB = np.median(cat_clusB.magnitude)


                    if np.abs(magA-magB) <= .3:


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
                                clusA = cat_clusA.Cluster.iloc[0]
                                cat_clusA = cat_clusA.append(cat_clusB)
                                cat_clusA.loc[:,'Cluster'] = clusA
                                print('Converging clusters',clusA,' and ',clusB,file=f)


                        clus_converge = clus_converge.append(cat_clusA).drop_duplicates('event_ID')    

        print('converged',file=f)
    return clus_converge





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

def RemoveMagnitudes(catalog, verbose = 1):
    """
    Removes earthquake events from a seismic catalog whose magnitudes are
    more than 0.3 above or below the median magnitude of their respective clusters.
    
    Parameters
    ----------
    catalog : pandas DataFrame
        The seismic catalog to process. Must contain at least the following columns:
            * Cluster: a numeric label indicating the cluster to which each event belongs
            * magnitude: the magnitude of each event
    verbose : int, optional
        Whether to print progress messages during processing. Set to 1 to enable
        (default) or 0 to disable.
    
    Returns
    -------
    pandas DataFrame
        The seismic catalog with the offending events removed.
    """
    cat_final = pd.DataFrame()


    for i, clus in enumerate(np.unique(catalog.Cluster)):
        if verbose:
        
            if i%100==0:
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


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################

def formatWS2021(pathCat):
    """
    Formats Waldhauser and Schaff 2021 catalog from a csv file located at pathCat into a new dataframe with additional columns 

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

def latlon2meter_vec(coords1, coords2):
    # Vectorize distance calculation using numpy broadcasting
    lat1, lon1 = np.radians(coords1)
    lat2, lon2 = np.radians(coords2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = 6371000 * c
    return distance


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


##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################


