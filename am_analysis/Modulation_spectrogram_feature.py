# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:28:56 2021

@author: Yi.Zhu
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:13:37 2021

@author: richa
"""

import numpy as np
import am_analysis as ama
from numba import jit
from skimage.util.shape import view_as_windows

@jit(nopython=True)
def segsum_matrix(mat,n_mbin,n_fbin,m_stride,f_stride,m_lo=0,m_hi=20,f_lo=0,f_hi=20):
    res = np.empty((n_mbin,n_fbin)).astype(np.float32)
    
    for f in np.arange(f_lo,f_hi):
        for m in np.arange(m_lo,m_hi):
            res[f,m] = np.sum(mat[f*f_stride:(f+1)*f_stride,m*m_stride:(m+1)*m_stride])
            
    return np.reshape(res,(n_mbin*n_fbin))

# original version of binned modulation spectrogram energies, might be slow to compute
def mspec_energy(signal,fs,mod_lim=20,freq_lim=8000,n_mod_bin=20,n_f_bin=20,win_size=256,fft_factor_y=4,fft_factor_x=4):
    # Compute modulation spectrogram
    mspec_data = ama.strfft_modulation_spectrogram(signal,
                                  fs=fs, 
                                  win_size=win_size, 
                                  win_shift=0.125*win_size, 
                                  fft_factor_y=fft_factor_y, 
                                  win_function_y='hamming', 
                                  fft_factor_x=fft_factor_x, 
                                  win_function_x='hamming', 
                                  channel_names=None)
    # Reshape power values of modulation spectrogram into 2-D
    MS = mspec_data['power_modulation_spectrogram']
    MS = MS[:,:,0]
    # Range of modulation frequency and conventional frequency
    mod_lim = mod_lim
    freq_lim = freq_lim
    # Convert Hz into step size for further summation (e.g. mod_step=2: 1Hz=2)
    mod_step = int((mod_lim/n_mod_bin)/mspec_data['freq_mod_delta'])
    freq_step = int((freq_lim/n_f_bin)/mspec_data['freq_delta'])
    
    # # Energies to be stored
    mod = segsum_matrix(MS,n_mod_bin,n_f_bin,mod_step,freq_step)
    return 10*np.log10(mod)


def chop_msr(msr,out_rg,fstep,mstep):
    return msr[int(out_rg[0][0]*fstep):int(out_rg[0][1]*fstep),int(out_rg[1][0]*mstep):int(out_rg[1][1]*mstep)]
    
def strided4D(arr,arr2,s):
    return view_as_windows(arr, arr2.shape, step=s)

def strided_conv(arr,kr,s):
    arr4D = strided4D(arr=arr,arr2=kr,s=s)
    return np.tensordot(arr4D,kr,axes=((2,3),(0,1)))

def msr_kernel(fstep,mstep):
    return np.ones((fstep,mstep),dtype=np.float32) # kernel elements are 1 -> sum up all values

# bin modulation spectrogram energies using only matrix operation
def mspec_energy2(signal,fs,mod_lim=(0,20),freq_lim=(0,8000),n_mod_bin=20,n_f_bin=20,win_size=256,fft_factor_y=4,fft_factor_x=4):
    # Compute modulation spectrogram
    mspec_data = ama.strfft_modulation_spectrogram(signal,
                                  fs=fs, 
                                  win_size=win_size, 
                                  win_shift=0.125*win_size, 
                                  fft_factor_y=fft_factor_y, 
                                  win_function_y='hamming', 
                                  fft_factor_x=fft_factor_x, 
                                  win_function_x='hamming', 
                                  channel_names=None)
    # Reshape power values of modulation spectrogram into 2-D
    MS = mspec_data['power_modulation_spectrogram']
    MS = MS[:,:,0].astype(np.float32)
    # Convert bin width (Hz) into step size (number of samples) for further summation 
    # E.g. mod_step=2 : 2-steps/mod_bin(=1Hz)
    mod_range = mod_lim[1] - mod_lim[0]
    freq_range = freq_lim[1] - freq_lim[0]
    mod_step = int((mod_range/n_mod_bin)/mspec_data['freq_mod_delta'])
    freq_step = int((freq_range/n_f_bin)/mspec_data['freq_delta'])
    # Keep only part of the MSR
    msr_cp = chop_msr(MS,((0,n_f_bin),(0,n_mod_bin)),freq_step,mod_step)
    msr_all = np.sum(msr_cp)
    # Segment and bin
    kr = msr_kernel(freq_step,mod_step) # kernel size
    msr_sg = strided_conv(msr_cp,kr,s=(freq_step,mod_step))
    
    return np.reshape(10*np.log10(msr_sg/msr_all),(400,))

def mspec_energy3(signal,fs, win_size=256,fft_factor_y=8,fft_factor_x=16):
    # Compute modulation spectrogram
    mspec_data = ama.strfft_modulation_spectrogram(signal,
                                  fs=fs, 
                                  win_size=win_size, 
                                  win_shift=0.125*win_size, 
                                  fft_factor_y=fft_factor_y, 
                                  win_function_y='hamming', 
                                  fft_factor_x=fft_factor_x, 
                                  win_function_x='hamming', 
                                  channel_names=None)
    # Reshape power values of modulation spectrogram into 2-D
    MS = mspec_data['power_modulation_spectrogram']
    MS = MS[:,:,0].astype(np.float32)
    
    return 10*np.log10(MS)

    
def get_mspec_descriptors(mod, mod_lim=20, freq_lim=8000, n_mod_bin=20, n_freq_bin=20):
    """

    Parameters
    ----------
    mod : 2D Numpy array
        Modulation spectrogram
    mod_lim : int
        Upper limit of modulation frequency. The default is 20.
    freq_lim : int
        Upper limit of frequency. The default is 8000.
    n_mod_bin : int, optional
        Number of modulation frequency bins. The default is 20.
    n_freq_bin : int, optional
        Number of frequency bins. The default is 20.

    Returns
    -------
    Modulation spectrogram descriptors: 1D numpy array

    """
    n_fea = 8 #Number of features to compute
    mod = 10**(mod/10) #Convert energies in dB to original values
    n_mod_bin = n_mod_bin #Number of modulation frequency bins
    n_freq_bin = n_freq_bin #Number of conventional frequency bins
    mod = np.reshape(mod,(n_freq_bin, n_mod_bin)) #Reshape psd matrix
    ds_mod = np.empty((n_mod_bin,n_fea))*np.nan #Initialize a matrix to store descriptors in all bins
    ds_freq = np.empty((n_freq_bin,n_fea))*np.nan
    
    def get_subband_descriptors(psd, freq_range):
        #Initialize a matrix to store features
        ft=np.empty((8))*np.nan
        lo,hi = freq_range[0], freq_range[-1]#Smallest and largest value of freq_range
        
        #Centroid
        ft[0] = np.sum(psd*freq_range)/np.sum(psd)
        #Entropy
        ft[1]=-np.sum(psd*np.log(psd))/np.log(hi-lo)
        #Spread
        ft[2]=np.sqrt(np.sum(np.square(freq_range-ft[0])*psd)/np.sum(psd))
        #skewness
        ft[3]=np.sum(np.power(freq_range-ft[0],3)*psd)/(np.sum(psd)*ft[2]**3)
        #kurtosis
        ft[4]=np.sum(np.power(freq_range-ft[0],4)*psd)/(np.sum(psd)*ft[2]**4)
        #flatness
        arth_mn=np.mean(psd)/(hi-lo)
        geo_mn=np.power(np.exp(np.sum(np.log(psd))),(1/(hi-lo)))
        ft[5]=geo_mn/arth_mn
        #crest
        ft[6]=np.max(psd)/(np.sum(psd)/(hi-lo))
        #flux
        ft[7]=np.sum(np.abs(np.diff(psd)))
        
        return ft
    
    #Loop through all modulation frequency bands
    freq_bin_width = freq_lim/n_freq_bin
    mod_bin_width = mod_lim/n_mod_bin
    freq = np.arange(0,freq_lim,freq_bin_width)+freq_bin_width/2 #List of center values of frequency bins
    mod_freq = np.arange(0,mod_lim,mod_bin_width)+mod_bin_width/2 #List of center values of modulation frequency bins
    #Calculate features for each modulation frequency bin
    for mod_band in np.arange(n_mod_bin):
        ds_mod[mod_band,:] = get_subband_descriptors(mod[:,mod_band], freq)
    #Calculate features for each conventional frequency bin
    for freq_band in np.arange(n_freq_bin):
        ds_freq[freq_band,:] = get_subband_descriptors(mod[freq_band,:], mod_freq)
    
    return np.concatenate((np.reshape(ds_mod, (8*n_mod_bin)), np.reshape(ds_freq, (8*n_freq_bin))),axis=None)


def extract_mod_fea(glued_audio_list,fs):
    """

    Parameters
    ----------
    glued_audio_list : list
        List of all audio signals
        shape: NUMBER OF INSTANCES * SIGNAL LENGTH
    fs : int
        Sampling rate

    Returns
    -------
    mod_fea : Numpy array
        Modulation spectrogram features, including modulation spectrogram energies
        and spectral descriptors

    """
    n_samples = len(glued_audio_list)
    n_fea = 8
    n_mod_bin = 20
    n_freq_bin = 20
    n_en_sample = n_mod_bin*n_freq_bin
    n_lld_sample = n_fea*n_mod_bin + n_fea*n_freq_bin
    
    n_fea = n_en_sample + n_lld_sample
    mod_fea = np.empty((n_samples,n_fea))*np.nan
    
    for i in range(n_samples):

        mod_fea[i,:n_en_sample] = mspec_energy2(glued_audio_list[i],fs=fs,freq_lim=(0,6000))
        mod_fea[i,n_en_sample:] = get_mspec_descriptors(mod_fea[i,:n_en_sample],
                                                                mod_lim=20,
                                                                freq_lim=6000,
                                                                n_mod_bin=n_mod_bin,
                                                                n_freq_bin=n_freq_bin)
    
    return mod_fea