'''
seed4 h5 builder
bandpass filtered: 1-45Hz bandpass + 50Hz notch, 200Hz downsampled.

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
HFREQ = 45 # eeg filter
LFREQ = 1 # eeg filter
RSFREQ = 200 # sampling
# trigger
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
start_point_list = [tup[0]*RSFREQ for tup in MATERIAL_TIMESTAMPS[exp_order]]
end_point_list = [tup[1]*RSFREQ for tup in MATERIAL_TIMESTAMPS[exp_order]]


####################
# create hdf5 file #
####################
with h5py.File(f'{dataFolder}/{args.name}(1-45+50).h5','w') as f:
    subgroup = f.create_group(args.name)
    subgroup.attrs['label'] = LABELS[exp_order]
    subgroup.attrs['trialStart'] = start_point_list
    subgroup.attrs['trialEnd'] = end_point_list
    subgroup.attrs['sFreq'] = RSFREQ
    subeeg = subgroup.create_dataset('eeg',data=raw[:][0])
    subeeg.attrs['chOrder'] = raw.info['ch_names']
    subeeg.attrs['hFreq'] = HFREQ
    subeeg.attrs['lFreq'] = LFREQ