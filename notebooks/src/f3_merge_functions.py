## spectrogram_functions.py
## For loading waveforms and generating spectrograms
## from spectrogram_functions import wf_to_H5, gen_sgram_QC_noAlias


import h5py
import numpy as np
import obspy
from obspy import read
import datetime

####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------




def addEventClusters(cat_add,cat_BXX4):
    """
    Add event clusters from cat_add to cat_BXX4 if they are not already present in cat_BXX4.

    Parameters:
    -----------
    cat_add: pandas.DataFrame
        A pandas DataFrame containing earthquake data, including columns for event ID and cluster ID.
    cat_BXX4: pandas.DataFrame
        A pandas DataFrame containing earthquake data, including columns for event ID and cluster ID.

    Returns:
    --------
    cat_BXX4: pandas.DataFrame
        A pandas DataFrame containing earthquake data from both cat_add and cat_BXX4, with clusters from cat_add appended
        to cat_BXX4 if they are not already present in cat_BXX4.
    """

    for i, ev in enumerate(cat_add.event_ID):


        if ev not in list(cat_BXX4.event_ID):
    #         print(ev)
            df = cat_add[cat_add.event_ID==ev]
            cl = df.Cluster.iloc[0]
            df_cl = cat_add[cat_add.Cluster==cl]   


            if any(i in list(cat_BXX4.event_ID) for i in list(df_cl.event_ID)):

                for ev in list(df_cl.event_ID):
                    try:

                        clus_new = cat_BXX4[cat_BXX4.event_ID==ev].Cluster.iloc[0]

                        df_cl.Cluster.iloc[:] = clus_new

                    except:
                        pass

            cat_BXX4 = cat_BXX4.append(df_cl)

    cat_BXX4.drop_duplicates(['event_ID'],inplace=True)
    return cat_BXX4