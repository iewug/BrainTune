'''
seed5 h5 builder
filtering information: 1Hz high-pass + 50Hz notch, 200Hz downsampling

hdf5 file - <subject's name>   - eeg (62,xxx)
            |- label             |- chOrder (62,)
            |- trialEnd #samples |- hFreq #45Hz
            |- trialStart #samples |- lFreq #1Hz
            |- sFreq #200 Hz
            |- score 
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
# trigger information
start_point_lists = np.array([[30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
                             [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
                             [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]])
end_point_lists = np.array([[102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
                           [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
                           [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]])
# label顺序. Disgust(0), Fear(1), Sad(2), Neutral(3), Happy(4)
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
raw.filter(LFREQ,None)
raw.resample(RSFREQ)


###########
# trigger #
###########
name_order, exp_order = args.name.split("_")
name_order = int(name_order)-1
exp_order = int(exp_order)-1
start_point_list = start_point_lists[exp_order] * RSFREQ
end_point_list = end_point_lists[exp_order] * RSFREQ


####################
# create hdf5 file #
####################
with h5py.File(f'{dataFolder}/{args.name}(1-_+50).h5','w') as f:
    subgroup = f.create_group(args.name)
    subgroup.attrs['label'] = label_lists[exp_order]
    subgroup.attrs['trialStart'] = start_point_list
    subgroup.attrs['trialEnd'] = end_point_list
    subgroup.attrs['sFreq'] = RSFREQ
    subgroup.attrs['score'] = score_lists[name_order][exp_order]
    subeeg = subgroup.create_dataset('eeg',data=raw[:][0])
    subeeg.attrs['chOrder'] = raw.info['ch_names']
    subeeg.attrs['hFreq'] = HFREQ
    subeeg.attrs['lFreq'] = LFREQ