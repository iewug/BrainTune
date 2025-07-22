import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from artifact.artifact import EEGArtifact
import argparse
import h5py


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="1_1", type=str, help='subject name')
args = parser.parse_args()


# load h5py
dataFolder = f'../../data/seed4/{args.name}'
f = h5py.File(f'{dataFolder}/{args.name}(1-45+50).h5','r')
f_hp = h5py.File(f'{dataFolder}/{args.name}(1-_+50).h5','r') # high pass


# build model
mymodel = EEGArtifact(
        f[args.name]['eeg'][:]*1e6, # V -> μV
        int(f[args.name].attrs['sFreq']),
        ch_order=f[args.name]['eeg'].attrs['chOrder'],
        trigger=None,
        highpass_raw_data=f_hp[args.name]['eeg'][:]*1e6,
        EOG_ch_prefix=['FP','AF']
    )


# save result
# mymodel.plot(save_dir=f'result/{args.name}')
mymodel.saveMask(save_dir=f'{dataFolder}/artifact')