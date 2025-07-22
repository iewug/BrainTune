'''
seed3 DE builder
raw cnt file, 0.5~70Hz bandpass, 50Hz notch, 200Hz resample
draw DE feaure, save in the 

hdf5 file - <subject's name>   - de (xxx,5,62)
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
# trigger info
start_point_list = np.array([27000,290000,551000,784000,1050000,1262000,1484000,1748000,1993000,2287000,2551000,2812000,3072000,3335000,3599000]) # 1000Hz
end_point_list = np.array([262000,523000,757000,1022000,1235000,1457000,1721000,1964000,2258000,2524000,2786000,3045000,3307000,3573000,3805000])
# label seq
label_list = np.array([2,1,0,0,1,2,0,1,2,2,1,0,1,2,0])


##########
# parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="1_1", type=str, help='subject name')
args = parser.parse_args()


#############
# load data #
#############
dataFolder = f'../../data/seed3/{args.name}'
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
start_point_list //= origin_freq
end_point_list //= origin_freq


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
    subgroup.attrs['label'] = label_list
    subgroup.attrs['trialStart'] = start_point_list
    subgroup.attrs['trialEnd'] = end_point_list
    subgroup.attrs['sFreq'] = RSFREQ
    subeeg = subgroup.create_dataset('de',data=de)
    subeeg.attrs['chOrder'] = raw.info['ch_names']
    subeeg.attrs['winSiz'] = WINSIZ
    subeeg.attrs['stepSiz'] = STEPSIZ
    subeeg.attrs['hFreq'] = HFREQ
    subeeg.attrs['lFreq'] = LFREQ
