'''
Artifact mask
'''
import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from utils import _mad, _mat_iqr

IQR_TO_SD = 0.7413
MAD_TO_SD = 1.4826

class EEGArtifact:
    '''
    Artifact detection is based on:
    - Outliers and constant signals
    - Abnormal amplitude (whether window data variance deviates from normal values)
    - High-frequency noise (whether window high-frequency data variance deviates from normal values)
    - Vertical eye movements

    Parameters:
    -----------
    - data: Band-pass filtered and notch-filtered EEG signals, in μV
    - freq: EEG signal frequency; high-frequency noise requires frequencies above 100Hz
    - highpass_raw_data: Raw EEG signals that have only undergone high-pass and notch filtering
    - ch_order: Channel order ['Fp1',...]
    - trigger: Trigger information {trigger:[sample_points]}
    - window_size: Window length (s)
    - step_size: Step size (s)
    - T_flat: Threshold for constant signal detection
    - T_range: Threshold for value range detection
    - T_hfzsd: Threshold for high-frequency noise detection
    - T_ampzsd: Threshold for abnormal amplitude detection
    - EOG_ch_prefix: Array of channel prefixes for EOG inspection
    - online: Online or offline version
                  Offline: Abnormal amplitude (temporal detection) and high-frequency noise use future data.
                  Online: Artifact calculation does not use future input data;
                          abnormal amplitude (temporal detection) and high-frequency noise differ slightly from the offline version.
    - iter_size: Calculate artifacts every `iter_size` seconds, simulating the online version.
    '''
    
    def __init__(self,
                 data:np.ndarray, # EEG data, band-pass and notch-filtered
                 freq:int, # frequency
                 highpass_raw_data:Optional[np.ndarray]=None, # Raw signals that have undergone high-pass and notch filtering
                 ch_order:Optional[List[str]]=None, # Channel order
                 trigger:Optional[Dict[str, List[int]]]=None, # e.g., {'1':[100,200]}
                 window_size=1, # Window length s
                 step_size=1, # Step size s
                 T_flat=1, # Constant signal
                 T_range=[-120,150], # Value range
                 T_hfzsd=8.0, # High-frequency noise
                 T_ampzsd=5.0, # Amplitude detection
                 EOG_ch_prefix:List[str]=["Fp", "AF"], # Array of channel prefixes for EOG inspection
                 online=True, # Online or offline version
                 iter_size=10 # Calculate artifacts every `iter_size` seconds, simulating the online version
                ) -> None:
        # get EEG data
        self.data = data
        self.freq = freq
        self.highpass_raw_data = highpass_raw_data
        self.ch_order = ch_order
        # Elements in trigger are sample points, need to convert to corresponding windows
        if trigger is not None:
            self.trigger = {key: [value // (step_size * freq) for value in values] for key, values in trigger.items()}
        else:
            self.trigger = None
        # get threshold settings
        self.window_size = window_size*self.freq
        self.step_size = step_size*self.freq
        self.T_flat = T_flat
        self.T_range = T_range
        self.T_hfzsd = T_hfzsd
        self.T_ampzsd = T_ampzsd
        
        # Number of sliding windows, requires complete window size
        window_num = (self.data.shape[1]-self.window_size)//self.step_size+1

        # Storage for results, shape (num_channels, num_windows), True for bad windows, False for good windows
        self.mask_flat, self.mask_naninf = [np.empty((self.data.shape[0], window_num), dtype=bool) for _ in range(2)]
        self.mask_amp_tem, self.mask_amp_spa = [np.empty((self.data.shape[0], window_num), dtype=bool) for _ in range(2)]
        self.mask_hfnoise = np.empty((self.data.shape[0], window_num), dtype=bool)
        self.mask_vertical_eog = np.empty((self.data.shape[0], window_num), dtype=bool)

        # EOG only checks Fp and AF channels
        eog_indices = None
        if self.ch_order is not None:
            eog_indices = []
            for idx, element in enumerate(ch_order):
                if any(element.startswith(prefix) for prefix in EOG_ch_prefix):
                    eog_indices.append(idx)

        # Calculate!
        rSDs = [] # rSD = MAD * MAD_TO_SD or IQR * IDR_TO_SD, assuming normality
        noisinesses = [] # method3-1
        for window_idx, i in enumerate(range(0,self.data.shape[1]-self.window_size+1,self.step_size)):
            # Slice
            window_data = self.data[:,i:i+self.window_size]

            # method1: Constant signals and outliers
            winmask_naninf, winmask_flat = self._detect_flat_nan_inf(window_data)
            self.mask_flat[:,window_idx] = winmask_flat
            self.mask_naninf[:,window_idx] = winmask_naninf
            # When analyzing with other methods, channels marked as bad by method1 are not considered.
            winmask_flat_naninf = winmask_flat | winmask_naninf

            # method2-1: Abnormal amplitude (temporal detection), store in array for later use
            rSDs.append(_mat_iqr(window_data,axis=1)*IQR_TO_SD)

            # method2-2: Abnormal amplitude (spatial detection)
            winmask_amp_spa = self._detect_abnormal_amplitude_spatial(window_data,winmask_flat_naninf)
            self.mask_amp_spa[:,window_idx] = winmask_amp_spa

            # method3: Excessive high-frequency signals
            if self.highpass_raw_data is not None and self.freq > 100:
                window_data_raw = self.highpass_raw_data[:,i:i+self.window_size]
                noisinesses.append(_mat_iqr(window_data_raw-window_data,axis=1) * IQR_TO_SD)

            # method4: vertical eog
            self.mask_vertical_eog[:,window_idx] = self._detect_vertical_eog(window_data,eog_index=eog_indices)

        self.mask_flat_naninf = self.mask_flat | self.mask_naninf
        rSDs = np.array(rSDs).T
        noisinesses = np.array(noisinesses).T
        if online:
            for i in range(0,window_num,iter_size):
                # method2-1 continued: Abnormal amplitude (temporal detection)
                self.mask_amp_tem[:,i:i+iter_size] = self._detect_abnormal_amplitude_temporal(rSDs[:,:i+iter_size],self.mask_flat_naninf[:,:i+iter_size])[:,i:i+iter_size]
                # method3 continued: Excessive high-frequency signals
                if self.highpass_raw_data is not None and self.freq > 100:
                    self.mask_hfnoise[:,i:i+iter_size] = self._detect_hfnoise(noisinesses[:,:i+iter_size],self.mask_flat_naninf[:,:i+iter_size])[:,i:i+iter_size]
        else:
            # method2-1 continued: Abnormal amplitude (temporal detection)
            self.mask_amp_tem = self._detect_abnormal_amplitude_temporal(rSDs,self.mask_flat_naninf)
            # method3 continued: Excessive high-frequency signals
            if self.highpass_raw_data is not None and self.freq > 100:
                self.mask_hfnoise = self._detect_hfnoise(noisinesses,self.mask_flat_naninf)
 

    def _detect_flat_nan_inf(self, window_data:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Detects constant signals and outliers in each channel, where:
        - Constant signal: MAD < self.T_flat or SD < self.T_flat
        - Outlier: inf, NaN, or outside self.T_range

        Parameters
        ----------
        window_data: Data for one window (ch_num, freq*window_size)

        Returns
        -------
        Boolean array for outliers, boolean array for constant signals, True for bad, False for good (ch_num,)
        '''
        flat_by_mad = _mad(window_data, axis=1) < self.T_flat
        flat_by_sd = np.std(window_data, axis=1) < self.T_flat
        bad_by_nan = np.isnan(window_data).any(axis=1)
        bad_by_inf = np.isinf(window_data).any(axis=1)
        exceeds_upper_threshold = (window_data > self.T_range[1]).any(axis=1)
        below_lower_threshold = (window_data < self.T_range[0]).any(axis=1)
        return bad_by_inf | bad_by_nan | exceeds_upper_threshold | below_lower_threshold, flat_by_mad | flat_by_sd 
    

    def _detect_abnormal_amplitude_spatial(self, window_data:np.ndarray, mask_bad:np.ndarray) -> np.ndarray:
        '''
        Detects whether each channel in a given window at the same time has an abnormally low or high amplitude compared to other channels.
        
        The formula for abnormal amplitude detection uses the robust z-score of robust standard deviation:

        ``rSD_zscore = (rSD-rSDs_median) / rSDs_rSD > T_zsd``.
        
        rSD: Robust standard deviation of a channel in this window, estimated using IQR; rSDs: Robust standard deviations of all channels; rSDs_rSD/median are the robust standard deviation or median of the robust standard deviations.

        Parameters
        ----------
        - window_data: Data for one window (ch_num, freq*window_size)
        - mask_bad: 1D boolean ndarray not included in the analysis, True for bad channels (ch_num,)

        Returns
        -------
        Boolean array for outliers, True for bad, False for good, mask_bad channels are False (ch_num,)
        '''
        # If number of good channels is less than 3, return all as bad
        if np.sum(mask_bad == False) < 3:
            return np.full(window_data.shape[0],True,dtype=bool)
        rSDs = _mat_iqr(window_data[~mask_bad],axis=1) * IQR_TO_SD # Robust standard deviation of all channels
        rSDs_rSD = _mat_iqr(rSDs) * IQR_TO_SD
        rSDs_median = np.median(rSDs)
        rSD_zscore = np.zeros(window_data.shape[0])
        rSD_zscore[~mask_bad] = (rSDs-rSDs_median)/rSDs_rSD
        return np.abs(rSD_zscore) > self.T_ampzsd


    def _detect_abnormal_amplitude_temporal(self, rSDs:np.ndarray, mask_bad:np.ndarray) -> np.ndarray:
        '''
        Detects whether a window has an abnormally low or high amplitude compared to all windows for that channel.

        Parameters
        ----------
        - rSDs: rSD for each channel (ch_num, time_len)
        - mask_bad: 2D boolean ndarray not included in the analysis, True for bad channels (ch_num, time_len)

        Returns
        -------
        Boolean array for outliers, True for bad, False for good, mask_bad channels are False (ch_num, time_len)
        '''
        # There seems to be no better slicing operation, so we analyze channel by channel here
        rSD_zscore = np.zeros_like(rSDs)
        for idx in range(rSDs.shape[0]): # For each channel
            mask_ch = ~mask_bad[idx] # Good windows
            rSDs_ch = rSDs[idx][mask_ch]
            if rSDs_ch.shape[0] <= 5: # Completely bad channel
                continue
            rSDs_ch_rSD = _mat_iqr(rSDs_ch) * IQR_TO_SD
            rSDs_ch_median = np.median(rSDs_ch)
            rSD_zscore[idx][mask_ch] = (rSDs_ch-rSDs_ch_median)/rSDs_ch_rSD
        return np.abs(rSD_zscore) > self.T_ampzsd


    def _detect_hfnoise(self, noisinesses:np.ndarray, mask_bad:np.ndarray) -> np.ndarray:
        '''
        Detects whether there is excessive high-frequency noise.

        Noisiness is defined as the robust standard deviation of the high-frequency component divided by the robust standard deviation of the low-frequency component. The robust z-score method is also used to determine if noisiness is too high.
        However, analyzing the high-frequency components of each channel might be more effective than analyzing the ratio, i.e., determining by the robust z-score of the high-frequency component's robust standard deviation.
        Unlike amplitude detection which uses row-wise and column-wise comparisons, high-frequency noise considers data from all channel windows together. This is mainly to avoid "false positives" when some channels usually have very low noise but suddenly have some large noise, and corresponding "false negatives".
        Furthermore, the Nyquist theorem requires the sampling rate to be more than twice the maximum effective value; to preserve more high-frequency noise, a necessary sampling rate is required.

        Parameters
        ----------
        - noisinesses: Noisiness for each channel (ch_num, time_len)
        - mask_bad: 2D boolean ndarray not included in the analysis, True for bad channels (ch_num, time_len)

        Returns
        -------
        Boolean array for outliers, True for bad, False for good, mask_bad channels are False (ch_num, time_len)
        '''
        noise_zscore = np.zeros_like(noisinesses)
        noise_median = np.median(noisinesses[~mask_bad])
        noise_rsd = _mat_iqr(noisinesses[~mask_bad]) * IQR_TO_SD
        noise_zscore[~mask_bad] = (noisinesses[~mask_bad]-noise_median)/noise_rsd
        return noise_zscore > self.T_hfzsd
    

    def _detect_vertical_eog(self, window_data:np.ndarray, segment_length = 0.08, threshold = 60, eog_index:Optional[List[str]]=None) -> np.ndarray:
        '''
        Checks for vertical electrooculogram (EOG) artifacts.
        
        Determines by checking if there is a continuous increase (>-10μV) lasting longer than segment_length (s), with a cumulative increase exceeding threshold (μV), and the final value also exceeding threshold.

        Parameters
        ----------
        - window_data: Data for one window (ch_num, freq*window_size)
        - eog_index: Only check selected channels

        Returns
        -------
        Boolean array for outliers, True for bad, False for good, non-eog_index channels are False (ch_num, time_len)
        '''
        num_channels, num_samples = window_data.shape
        result = np.zeros(num_channels, dtype=bool)
        segment_length = int(segment_length*self.freq)
        differences = np.diff(window_data,axis=1)
        for channel in range(num_channels):
            # Only consider channels in eog_index
            if eog_index is not None:
                if channel not in eog_index:
                    continue
            for i in range(num_samples - segment_length):
                segment_diff = differences[channel, i:i+segment_length]
                if np.all(segment_diff > -10) and (window_data[channel,i+segment_length]-window_data[channel,i]) > threshold and window_data[channel,i+segment_length]>threshold:
                    result[channel] = True
                    break
        return result
        # The simplest VEOG detection can be done by setting a threshold
        # return (window_data > 80).any(axis=1)


    def plot(self, save_dir='result', name:Optional[str]=None) -> None:
        '''
        Plots all masks, with one image per mask.

        Parameters
        ----------
        - save_dir: Save directory
        - name: Name of the saved plot
        '''
        os.makedirs(save_dir,exist_ok=True)
        masks = [self.mask_flat_naninf,self.mask_amp_tem,self.mask_amp_spa,self.mask_hfnoise,self.mask_vertical_eog]
        titles = ['flat_naninf','amp_tem','amp_spa','hfnoise','vertical_eog']

        for mask,title in zip(masks,titles):
            plt.figure(figsize=(16,8))
            plt.imshow(mask,cmap='gray',interpolation='nearest',aspect='auto')
            plt.title(f'{title}\nblack: good; white: bad')
            if self.ch_order is not None:
                # If channel order is provided, replace y-axis during plotting
                plt.yticks(np.arange(len(self.ch_order)), self.ch_order, fontsize=10)
            if self.trigger is not None:
                # If trigger information is provided:
                # 1. Add vertical lines during plotting
                for positions in self.trigger.values():
                    for pos in positions:
                        plt.axvline(x=pos, color='red', linestyle='--')         
                # 2. Mark these positions while retaining other x-axis ticks
                # Get current x-axis ticks
                current_ticks = plt.xticks()[0][2:-1] # Remove start and end
                # Add all positions from the dictionary to current ticks and set labels
                all_positions = np.concatenate(list(self.trigger.values()))
                new_ticks = np.unique(np.append(current_ticks, all_positions))
                new_labels = [str(int(tick)) for tick in new_ticks]
                # Update labels based on dictionary
                for label, positions in self.trigger.items():
                    for pos in positions:
                        new_labels[new_ticks.tolist().index(pos)] = label
                # Set new x-axis ticks and labels
                plt.xticks(new_ticks, new_labels)
            
            # save
            if name is None:
                save_path = os.path.join(save_dir,title)
            else:
                save_path = os.path.join(save_dir,f'{title}_{name}')
            plt.savefig(save_path)
            plt.close()


    def saveMask(self, save_dir='result', name:Optional[str]=None) -> None:
        '''
        Saves the masks as npy files.

        Parameters
        ----------
        - save_dir: Save directory
        - name: Name of the saved file
        '''
        os.makedirs(save_dir,exist_ok=True)
        masks = [self.mask_flat_naninf,self.mask_amp_tem,self.mask_amp_spa,self.mask_hfnoise,self.mask_vertical_eog]
        masks.append(np.logical_or.reduce(masks))
        titles = ['flat_naninf','amp_tem','amp_spa','hfnoise','vertical_eog','all']

        for mask, title in zip(masks, titles):
            if name is None:
                save_path = os.path.join(save_dir,f'{title}.npy')
            else:
                save_path = os.path.join(save_dir,f'{title}_{name}.npy')
            np.save(save_path,mask)


    

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="1_1", type=str, help='subject name')
    args = parser.parse_args()

    # load h5py
    dataFolder = f'../data/seed3/{args.name}'
    f = h5py.File(f'{dataFolder}/{args.name}(1-45+50).h5','r')
    f_hp = h5py.File(f'{dataFolder}/{args.name}(1-_+50).h5','r') # high pass

    # build model
    mymodel = EEGArtifact(
            f[args.name]['eeg'][:]*1e6, # V -> μV
            int(f[args.name].attrs['sFreq']),
            ch_order=f[args.name]['eeg'].attrs['chOrder'],
            trigger=None,
            highpass_raw_data=f_hp[args.name]['eeg'][:]*1e6,
            EOG_ch_prefix=['FP','AF'],
            online=True
        )
    # plot
    mymodel.plot(name=args.name,save_dir=f'result/{args.name}')
    mymodel.saveMask(name=args.name,save_dir=f'result/{args.name}')