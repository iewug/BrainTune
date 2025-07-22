'''
seed5 DE builder
Original CNT file: 0.5~70Hz bandpass filtered, 50Hz notch filtered, 200Hz downsampled.
Extracts Differential Entropy (DE) features and saves them in HDF5 format in the same directory as the original file.

hdf5 file - <subject's name>   - de (xxx,5,62)
            |- trialStart #s     |- lFreq #0.5Hz
            |- trialEnd #s       |- hFreq #70Hz
            |- sFreq #200Hz      |- winSiz #1s
            |- label             |- stepSiz #1s
            |- score             |- chOrder #List[str]
'''
import numpy as np
import scipy.signal as signal
import argparse
import mne
import h5py


##################
# get DE feature #
##################
def _get_average_psd(energy_graph, freq_bands, sample_rate, stft_n=256):
    start_index = int(np.floor(freq_bands[0] / sample_rate * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_rate * stft_n))
    ave_psd = np.mean(energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
    return ave_psd

def get_psd_feature(eeg: np.ndarray, sample_rate: int, window_size: int, stride_size:int, stft_n=256,
            freq_bands=[[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]) -> np.ndarray:
    """
    Extracts PSD features from a segment of time-series signal.
    :param np.ndarray eeg: Signal (n_channels, n_samples)
    :param int sample_rate: Sampling rate
    :param int window_size: Window length (s)
    :param int stride_size: Stride length (s)
    :param int stft_n: FFT parameter, defaults to 256
    :param list freq_bands: List of frequency band ranges, defaults to [[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]
    :return np.ndarray: PSD features (n_windows, n_freq_bands, n_channels)
    """
    n_channels, n_samples = eeg.shape
    point_per_window = int(sample_rate * window_size)
    point_per_stride = int(sample_rate * stride_size)
    window_num = int((n_samples-point_per_window) // point_per_stride) + 1
    psd = np.zeros((window_num, len(freq_bands), n_channels))
    for window_index in range(window_num):
        start_index, end_index = point_per_stride * window_index, point_per_stride * window_index + point_per_window
        window_data = eeg[:, start_index:end_index]
        hdata = window_data * signal.windows.hann(point_per_window)
        fft_data = np.fft.fft(hdata, n=stft_n)
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
        for band_index, band in enumerate(freq_bands):
            band_ave_psd = _get_average_psd(energy_graph, band, sample_rate, stft_n)
            psd[window_index, band_index, :] = band_ave_psd
    return psd

def get_de_feature(eeg: np.ndarray, sample_rate: int, window_size: int, stride_size: int, stft_n=256,
            freq_bands=[[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]) -> np.ndarray:
    """
    Extracts Differential Entropy (DE) features from time-series signal.
    :param np.ndarray eeg: Signal (n_channels, n_samples)
    :param int sample_rate: Sampling rate
    :param int window_size: Window length (s)
    :param int stride_size: Stride length (s)
    :param int stft_n: FFT parameter, defaults to 256
    :param list freq_bands: List of frequency band ranges, defaults to [[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]
    :return np.ndarray: DE features (n_windows, n_freq_bands, n_channels)
    """
    psd = get_psd_feature(eeg, sample_rate, window_size, stride_size, stft_n, freq_bands)
    de = np.log2(100*psd)
    return de


############
# settings #
############
HFREQ = 70 # eeg filter
LFREQ = 0.5 # eeg filter
RSFREQ = 200 # sampling
WINSIZ = 1 # extract de
STEPSIZ = 1 # extract de
# trigger information
start_point_lists = np.array([[30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
                             [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
                             [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]])
end_point_lists = np.array([[102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
                           [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
                           [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]])
# label sequence. Disgust(0), Fear(1), Sad(2), Neutral(3), Happy(4)
label_lists = np.array([[4,1,3,2,0,4,1,3,2,0,4,1,3,2,0],
                       [2,1,3,0,4,4,0,3,2,1,3,4,1,2,0],
                       [2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]])
# Evoked emotions Scores
score_lists = np.load('../../data/seed5/Scores.npy') # (num_subjects, num_experiments, num_videos)


##########
# parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="1_1", type=str, help='subject name')
args = parser.parse_args()


#############
# load data #
#############
dataFolder = f'../../data/seed5/{args.name}'
raw = mne.io.read_raw_cnt(f'{dataFolder}/{args.name}.cnt', data_format='int32', date_format='dd/mm/yy', preload=True)
useless_channels = ['VEO', 'HEO', 'M1', 'M2']
raw = raw.drop_channels(useless_channels)
raw.notch_filter(50)
raw.filter(LFREQ,HFREQ)
raw.resample(RSFREQ)


###########
# trigger #
###########
name_order, exp_order = args.name.split("_")
name_order = int(name_order)-1
exp_order = int(exp_order)-1
start_point_list = start_point_lists[exp_order]
end_point_list = end_point_lists[exp_order]


############
# eeg data #
############
dset = raw[:][0]
de = get_de_feature(dset,RSFREQ,WINSIZ,STEPSIZ)
# de = de.reshape(-1,5*32) # flatten


####################
# create hdf5 file #
####################
with h5py.File(f'{dataFolder}/de.h5','w') as f:
    subgroup = f.create_group(args.name)
    subgroup.attrs['label'] = label_lists[exp_order]
    subgroup.attrs['trialStart'] = start_point_list
    subgroup.attrs['trialEnd'] = end_point_list
    subgroup.attrs['sFreq'] = RSFREQ
    subgroup.attrs['score'] = score_lists[name_order][exp_order]
    subeeg = subgroup.create_dataset('de',data=de)
    subeeg.attrs['chOrder'] = raw.info['ch_names']
    subeeg.attrs['winSiz'] = WINSIZ
    subeeg.attrs['stepSiz'] = STEPSIZ
    subeeg.attrs['hFreq'] = HFREQ
    subeeg.attrs['lFreq'] = LFREQ
