'''
seed4 buildh5.py test file
'''
import h5py
import argparse


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
f = h5py.File(f"{dataFolder}/{args.name}(1-_+50).h5","r")
print(f[args.name].attrs['label'])
print(f[args.name].attrs['trialStart'])
print(f[args.name].attrs['trialEnd'])
print(f[args.name].attrs['sFreq'])
print(f[args.name]['eeg'][:].shape) # eeg data shape (62,xxx) 
print(f[args.name]['eeg'].attrs['chOrder']) # (62,)
print(f[args.name]['eeg'].attrs['hFreq'])
print(f[args.name]['eeg'].attrs['lFreq'])