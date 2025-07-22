'''
seed3 h5 builder
1Hz high pass + 50Hz notch, 200Hz resample

hdf5 file - <subject's name>   - eeg (62,xxx)
            |- label             |- chOrder (62,)
            |- trialEnd          |- hFreq #45Hz
            |- trialStart        |- lFreq #1Hz
            |- sFreq #200 Hz
'''
import mne
import h5py
import argparse
import numpy as np


############
# settings #
############
HFREQ = -1 # eeg filter
LFREQ = 1 # eeg filter
RSFREQ = 200 # sampling
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
raw.filter(LFREQ,None)
origin_freq = int(raw.info['sfreq'])
raw.resample(RSFREQ)


###########
# trigger #
###########
start_point_list //= (origin_freq//RSFREQ)
end_point_list //= (origin_freq//RSFREQ)


####################
# create hdf5 file #
####################
with h5py.File(f'{dataFolder}/{args.name}(1-_+50).h5','w') as f:
    subgroup = f.create_group(args.name)
    subgroup.attrs['label'] = label_list
    subgroup.attrs['trialStart'] = start_point_list
    subgroup.attrs['trialEnd'] = end_point_list
    subgroup.attrs['sFreq'] = RSFREQ
    subeeg = subgroup.create_dataset('eeg',data=raw[:][0])
    subeeg.attrs['chOrder'] = raw.info['ch_names']
    subeeg.attrs['hFreq'] = HFREQ
    subeeg.attrs['lFreq'] = LFREQ