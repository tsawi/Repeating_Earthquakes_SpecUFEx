## spectrogram_functions.py
## For loading waveforms and generating spectrograms
## from spectrogram_functions import wf_to_H5, gen_sgram_QC_noAlias


import h5py
import numpy as np
import obspy
from obspy import read
import datetime
import os 
import scipy.signal as sp

####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------






def load_wf(filename, lenData, channel_ID=None, verbose=0):
    """Loads a waveform file and returns the data.

    Arguments
    ---------
    filename: str
        Filename to load
    lenData: int
        Number of samples in the file. Must be uniform for all files.
    channel_ID: int
        If the fileis an obspy stream, this is the desired channel.
    """
    
    if ".txt" in filename:
        data = np.loadtxt(filename)
    else: #catch loading errors
        try:
            st = obspy.read(filename)
            ### REMOVE RESPONSE ??
            st.detrend('demean')
            data = st[channel_ID].data

            #make sure data same length
            Nkept = 0
            if len(data)==lenData:
                return data
            #Parkfield is sometimes one datapoint off
            elif np.abs(len(data) - lenData) ==1:
                data = data[:-1]
                Nkept += 1
                return data

            else:
                if verbose:
                    print(filename, ': data wrong length')
                    print(f"this event: {len(data)}, not {lenData}")
                pass
            
        except:
            print('loading error', filename)
            pass
        
    return
    

####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------


def wf_to_H5(pathProj,dataH5_path,pathWF,cat,lenData,channel_ID,station,channel,t0=0,t1=-1,verbose=0):
    """
    Load waveforms and store as arrays in H5 file
    (Does not store catalog -- that should be a different function)
    Returns the event IDs of successfully loaded waveforms

    Parameters
    ----------
    pathProj : str
        .
    dataH5_path : str
        .
     cat : pandas Dataframe
        Has columns called "ev_ID" and "filename".
    lenData : int
        In samples.
    channel_ID : int
        For obspy stream object.
    station : str
        .
    channel : str
        ex. N, EHZ.
    t0 : int
        Index of wf starting sample
    t1 : int
        Index of wf ending sample
        

    Returns
    -------
    evID_keep : list of str
        Ev_IDs of successfully loaded waveforms.
    data_trim : array
        waveform.
    """
   
###     clear old H5 if it exists, or else error will appear
    if os.path.exists(dataH5_path):
        os.remove(dataH5_path)
        

        
    
    evID_keep = []
    



    with h5py.File(dataH5_path,'a') as h5file:
    
    
        h5file.create_group("waveforms")
        h5file.create_group(f"waveforms/{station}")
        channel_group = h5file.create_group(f"waveforms/{station}/{channel}")
    
    
    
        dupl_evID = 0
        n=0
    
        for n, ev in cat.iterrows():
            if n%1000==0:
                print(n, '/', len(cat))
            data = load_wf(pathWF+ev["filename"], lenData, channel_ID)
            if data is not None:
                
                data_trim = data[t0:t1]
                
                channel_group.create_dataset(name=str(ev["ev_ID"]), data=data_trim)
                evID_keep.append(ev["ev_ID"])
            else:
                if verbose:
                    print(ev.ev_ID, " not saved")
    
 
    print(dupl_evID, ' duplicate events found and avoided')
    print(len(evID_keep), ' waveforms loaded')
    
    
    
    return evID_keep, data_trim
    
####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------
####-------------------------------------------------------------------------------

def gen_sgram_QC_noAlias(decimation_factor,key,evID_list,dataH5_path,trim=True,trimTime=False,saveMat=False,sgramOutfile='.',**args):
    """
    
    Generates spectrograms for a list of event IDs and performs quality control.

    Args:
        decimation_factor (int): The decimation factor to apply to the input signal.
        evID_list (list): A list of event IDs to process.
        dataH5_path (str): The path to the H5 data file containing the waveform data.
        trim (bool): If True, trims the spectrogram to specified frequency and time ranges.
        **args: Optional keyword arguments. Must include:
            - fs (int): The sampling rate of the input signal.
            - nperseg (int): The length of each segment.
            - nfft (int): The length of the FFT used. Zero padding is used if a larger FFT is desired.
            - mode (str): The FFT mode. Options are 'complex', 'magnitude', 'angle', 'phase'.
            - scaling (str): The scaling mode for the spectrogram. Options are 'density', 'spectrum'.
            - fmin (float): The minimum frequency to include in the spectrogram.
            - fmax (float): The maximum frequency to include in the spectrogram.
            - tmin (float): The minimum time to include in the spectrogram.
            - tmax (float): The maximum time to include in the spectrogram.
            - station (str): The station code for the input signal.
            - channel (str): The channel code for the input signal.

    Yields:
        A tuple containing the following elements:
        - evID (str): The event ID.
        - STFT (ndarray): The spectrogram data.
        - fSTFT (ndarray): The frequency values for the spectrogram.
        - tSTFT (ndarray): The time values for the spectrogram.
        - normConstant (float): The normalization constant used for the spectrogram.
        - Nkept (int): The number of spectrograms that passed quality control.
        - evID_BADones (list): A list of event IDs that failed quality control.
        - i (int): An index used for tracking progress.

    Raises:
        KeyError: If any of the required keyword arguments are missing
        
    """
    fs=sr=args['fs']
    nperseg=args['nperseg']
#     noverlap=args['noverlap']
    nfft=args['nfft']
    mode=args['mode']
    scaling=args['scaling']
    fmin=args['fmin']
    fmax=args['fmax']
    tmin=args['tmin']
    tmax=args['tmax']
    
    Nkept = 0
    evID_BADones = []
    for i, evID in enumerate(evID_list):

        if i%1000==0:
            print(i,'/',len(evID_list))

        with h5py.File(dataH5_path,'a') as fileLoad:
            stations=args['station']
            data = fileLoad[f"waveforms/{stations}/{args['channel']}"].get(str(evID))[:]

        decimation_factor = int(decimation_factor) # has to be an int
        stft_time_step = 1./sr # same as input time series `x`
        noverlap = args.get('nperseg', 256) - 1
        
        
        
        fSTFT, tSTFT, STFT_raw = sp.spectrogram(x=data,
                                                    fs=fs,
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    #nfft=Length of the FFT used, if a zero padded FFT is desired
                                                    nfft=nfft,
                                                    scaling=scaling,
                                                    axis=-1,
                                                    mode=mode)

        if trim:
            freq_slice = np.where((fSTFT >= fmin) & (fSTFT <= fmax))
            #  keep only frequencies within range
            fSTFT   = fSTFT[freq_slice]
            STFT_0f = STFT_raw[freq_slice,:][0]
            
            
            time_slice = np.where((tSTFT >= tmin) & (tSTFT <= tmax))
            tSTFT   = tSTFT[time_slice]
            STFT_0 = STFT_0f[:,time_slice[0]]

        
        
        else:
            STFT_0 = STFT_raw
            # print(type(STFT_0))
            
            
        ### Average pooling
        trimmed_length = (STFT_0.shape[-1]//decimation_factor)*decimation_factor
        shape_for_pooling = tuple(STFT_0.shape[:-1]) + tuple((-1, decimation_factor))
        STFT_0 = np.mean(STFT_0[..., :trimmed_length].reshape(shape_for_pooling), axis=-1)
        tSTFT = tSTFT[:trimmed_length][::decimation_factor]            


        # =====  [BH added this, 10-31-2020]:
        # Quality control:
        if np.isnan(STFT_0).any()==1 or  np.median(STFT_0)==0 :
            if np.isnan(STFT_0).any()==1:
                print('OHHHH we got a NAN here!')
                #evID_list.remove(evID_list[i])
                evID_BADones.append(evID)
                pass
            if np.median(STFT_0)==0:
                print('OHHHH we got a ZERO median here!!')
                #evID_list.remove(evID_list[i])
                evID_BADones.append(evID)
                pass

        if np.isnan(STFT_0).any()==0 and  np.median(STFT_0)>0 :

            normConstant = np.median(STFT_0)

            STFT_norm = STFT_0 / normConstant  ##norm by median

            STFT_dB = 20*np.log10(STFT_norm, where=STFT_norm != 0)  ##convert to dB
            # STFT_shift = STFT_dB + np.abs(STFT_dB.min())  ##shift to be above 0
    #

            STFT = np.maximum(0, STFT_dB) #make sure nonnegative


            if  np.isnan(STFT).any()==1:
                print('OHHHH we got a NAN in the dB part!')
                evID_BADones.append(evID)
                pass
            # =================save .mat file==========================

            else:

                Nkept +=1

                if saveMat==True:
                    if not os.path.isdir(sgramOutfile):
                        os.mkdir(sgramOutfile)


                    spio.savemat(sgramOutfile + str(evID) + '.mat',
                              {'STFT':STFT,
                                'fs':fs,
                                'nfft':nfft,
                                'nperseg':nperseg,
                                'noverlap':noverlap,
                                'fSTFT':fSTFT,
                                'tSTFT':tSTFT})


            # print(type(STFT))

            yield evID,STFT,fSTFT,tSTFT, normConstant, Nkept,evID_BADones, i