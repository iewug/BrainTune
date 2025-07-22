'''
seed4 buildDE.py test file
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
f = h5py.File(f"{dataFolder}/de.h5","r")
print(f[args.name].attrs['label'])
print(f[args.name].attrs['trialStart'])
print(f[args.name].attrs['trialEnd'])
print(f[args.name].attrs['sFreq'])
print(f[args.name]['de'][:].shape) # eeg data shape (len,5,ch_num) 
print(f[args.name]['de'].attrs['chOrder']) # (ch_num,)
print(f[args.name]['de'].attrs['hFreq'])
print(f[args.name]['de'].attrs['lFreq'])
print(f[args.name]['de'].attrs['winSiz'])
print(f[args.name]['de'].attrs['stepSiz'])