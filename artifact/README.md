# EEG Artifact Identification and Visualization

![](./img/vis.gif)

## 0. Environment Setup

```bash
pip install numpy
pip install matplotlib
pip install h5py
```

## 1. Usage Instructions

- artifact.py: Inputs EEG NumPy arrays and other information, outputs artifact mask .npy files and images.

- drawEEG.py: Inputs EEG and artifact mask NumPy arrays, provides an EEG visualization interface with corresponding artifact masks.

## 2. Examples

- artifact.py
  - Requires EEG input in μV.

  - The two data inputs can be considered as:

    - 1-45Hz band-pass + 50Hz notch filtered

    - 1Hz high-pass + 50Hz notch filtered (for highpass_raw_data)

  - Detection conditions include: outliers and constant signals, abnormal amplitude, high-frequency noise (muscle artifacts), and vertical eye movement.

  - If detecting high-frequency noise, the sampling rate should not be too low.

  - Provides online and offline analysis versions, set via the online parameter.

```python
mymodel = EEGArtifact(
            f[args.name]['eeg'][:]*1e6, # V -> μV, 1-45+50 filtered
            f[args.name].attrs['sFreq'], # 200
            ch_order=f[args.name]['eeg'].attrs['chOrder'], # ['ch1','ch2',]
            highpass_raw_data=f_hp[args.name]['eeg'][:]*1e6, # 1-_+50 filtered
            online=True # Artifact detection for the current window does not use future data
        )
mymodel.plot(name=args.name,save_dir=f'result/{args.name}')
mymodel.saveMask(name=args.name,save_dir=f'result/{args.name}')
```

- drawEEG.py

  - Not designed as an object-oriented approach.

  - Requires EEG input in μV.

  - Adjustable parameters include:

    - n_display_channels = 10  # Number of channels to display at a time
    - n_display_windows = 10 # Number of seconds to display at a time
    - display_channel_width = 150 # Display width per channel (μV)
    - mask_window_sz = 1 # The length of a non-overlapping window represented by one element in the mask (s)

  - Furthermore, it is easy to modify this code to visualize EEG directly without drawing artifacts.

