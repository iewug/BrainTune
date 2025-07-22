'''
seed4 DE builder
Original CNT file: 0.5~70Hz bandpass filtered, 50Hz notch filtered, 200Hz downsampled.
Extracts Differential Entropy (DE) features and saves them in HDF5 format in the same directory as the original file.

hdf5 file - <subject's name>   - de (xxx//RSFREQ,5,62)
            |- trialStart #s     |- lFreq #0.5Hz
            |- trialEnd #s       |- hFreq #70Hz
            |- sFreq #200Hz      |- winSiz #1s
            |- label             |- stepSiz #1s
                                 |- chOrder #List[str]
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
MATERIAL_TIMESTAMPS = [
    [(20  , 199), (239 , 336), (376 , 616), (656 , 795), (835 , 932), (972 , 1139),
    (1179, 1299), (1339, 1583), (1623, 1770), (1810, 2043), (2083, 2152), (2192, 2334),
    (2374, 2615), (2655, 2817), (2857, 3155), (3195, 3363), (3409, 3511), (3551, 3742),
    (3782, 3948), (3988, 4060), (4100, 4218), (4258, 4389), (4429, 4605), (4645, 4809)],

    [(20,288), (328,445), (485,660), (700,874), (914,1152), (1192,1322), (1362,1514),
    (1554,1762), (1802,1952), (1992,2110), (2150,2427), (2467,2529), (2569,2735), (2775,2889),
    (2929,3125), (3165,3251), (3291,3577), (3617,3836), (3876,4065), (4105,4159), (4199,4361),
    (4401,4621), (4661,4781), (4821,4933)],

    [(20,247), (287,427), (467,570), (610,849), (889,1097), (1137,1266), (1306,1359),
    (1399,1521), (1561,1695), (1735,1803), (1843,2108), (2148,2321), (2361,2565), 
    (2605,2695), (2735,2870), (2910,3099), (3139,3217), (3257,3362), (3402,3590),
    (3630,3733), (3773,3947), (3987,4155), (4195,4389), (4429,4598)]
] # second
# label sequence
LABELS = np.array([[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
                   [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
                   [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]])


##########
# parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="1_1", type=str, help='subject name')
args = parser.parse_args()


#############
# load data #
#############
dataFolder = f'../../data/seed4/{args.name}'
raw = mne.io.read_raw_cnt(f'{dataFolder}/{args.name}.cnt', data_format='int32', date_format='dd/mm/yy', preload=True)
useless_channels = ['VEO', 'HEO', 'M1', 'M2']
raw = raw.drop_channels(useless_channels)
raw.notch_filter(50)
raw.filter(LFREQ,HFREQ)
origin_freq = int(raw.info['sfreq'])
raw.resample(RSFREQ)


###########
# trigger #
###########
exp_order = int(args.name[-1])-1
start_point_list = [tup[0] for tup in MATERIAL_TIMESTAMPS[exp_order]]
end_point_list = [tup[1] for tup in MATERIAL_TIMESTAMPS[exp_order]]


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
    subgroup.attrs['label'] = LABELS[exp_order]
    subgroup.attrs['trialStart'] = start_point_list
    subgroup.attrs['trialEnd'] = end_point_list
    subgroup.attrs['sFreq'] = RSFREQ
    subeeg = subgroup.create_dataset('de',data=de)
    subeeg.attrs['chOrder'] = raw.info['ch_names']
    subeeg.attrs['winSiz'] = WINSIZ
    subeeg.attrs['stepSiz'] = STEPSIZ
    subeeg.attrs['hFreq'] = HFREQ
    subeeg.attrs['lFreq'] = LFREQ
