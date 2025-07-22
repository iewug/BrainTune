'''
Visualize EEG, with bad channels marked by background color
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
import h5py


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', default="1_1", type=str, help='subject name')
args = parser.parse_args()


# EEG data
f = h5py.File(f'../data/seed3/{args.name}/{args.name}(1-45+50).h5',"r")
# f_raw = h5py.File(f'../data/{args.name}/{args.name}(1-_+50).h5',"r")
eeg_data = (f[args.name]['eeg'][:])*1e6 # μV
n_channels, n_timepoints = eeg_data.shape
channel_names = f[args.name]['eeg'].attrs['chOrder']
freq = int(f[args.name].attrs['sFreq'])
# Mask data
mask_load_folder = f'../data/seed5/{args.name}/artifact'
mask_flat = np.load(f'{mask_load_folder}/flat_naninf.npy')
mask_amp_spa = np.load(f'{mask_load_folder}/amp_spa.npy')
mask_amp_tem = np.load(f'{mask_load_folder}/amp_tem.npy')
mask_hf = np.load(f'{mask_load_folder}/hfnoise.npy')
mask_veog = np.load(f'{mask_load_folder}/vertical_eog.npy')
masks = [mask_flat,mask_amp_spa,mask_amp_tem,mask_hf,mask_veog]
mask_labels = ['flat,inf','amp_spa','amp_tem','hf(emg)','veog']
mask_colors = ['red','orange','yellow','green','blue','purple','pink']


# Parameters
n_display_channels = 10  # Number of channels to display at a time
n_display_windows = 10 # Number of seconds to display at a time
display_channel_width = 150 # Channel width (μV)
mask_window_sz = 1 # The length of a non-overlapping window represented by one element in the mask (s)

# Initialize displayed channels and time range
current_channels = list(range(n_display_channels))[::-1] # [9,8,...,0]
current_time = [0, n_display_windows*freq]  # Initial displayed sample point range [0,1000]

# Create figure and subplots
fig, ax = plt.subplots(figsize=(15, 10))
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.85)

# Create custom legend
patches = [mpatches.Patch(color=mask_colors[i], alpha=0.3, label=mask_labels[i]) for i in range(5)]


# Function to plot EEG
def plot_eeg():
    ax.clear()
    offset = np.arange(len(current_channels)) * display_channel_width # Channels are separated by display_channel_width μV
    for i, ch in enumerate(current_channels):
        ax.plot(np.arange(current_time[0], current_time[1])/freq, eeg_data[ch, current_time[0]:current_time[1]] + offset[i], color='black', label=channel_names[ch], linewidth=0.5)
        # Apply background color
        for color_idx, mask in enumerate(masks):
            for mask_idx in range(current_time[0]//(mask_window_sz*freq),current_time[1]//(mask_window_sz*freq)+1,1):
                if mask_idx < mask.shape[1] and mask[ch][mask_idx]: # If bad window. mask_idx needs to be within bounds as mask.shape[1] is actually shorter than EEG data
                    rect_start = max(current_time[0]/freq, mask_idx*mask_window_sz)
                    rect_end = min(current_time[1]/freq, (mask_idx+1)*mask_window_sz)
                    ax.fill_betweenx(y=[offset[i] - display_channel_width//2+display_channel_width//len(mask_labels)*color_idx, offset[i] - display_channel_width//2 + display_channel_width//len(mask_labels)*(color_idx+1)], x1=rect_start, x2=rect_end, color=mask_colors[color_idx], alpha=0.3, edgecolor='none')
    ax.set_yticks(offset)
    ax.set_yticklabels([channel_names[ch] for ch in current_channels])
    ax.set_xlim(current_time[0]/freq, current_time[1]/freq)
    ax.set_ylim(-display_channel_width//2, (n_display_channels - 1) * display_channel_width + display_channel_width//2)
    ax.set_xlabel("Time (s)")
    ax.set_title("EEG Data")
    # Add legend and set its position to upper right
    ax.legend(handles=patches, loc='upper right')
    plt.draw()

# Initial plot
plot_eeg()

# Create sliders
axcolor = 'lightgoldenrodyellow'
ax_time = plt.axes([0.1, 0.1, 0.7, 0.03], facecolor=axcolor)
ax_channel = plt.axes([0.9, 0.2, 0.03, 0.7], facecolor=axcolor)

time_slider = Slider(ax_time, 'Time', 0, n_timepoints - n_display_windows*freq, valinit=0, valstep=1)
channel_slider = Slider(ax_channel, 'Channels', 0, n_channels - n_display_channels, valinit=0, valstep=1, orientation='vertical')
# Sliders do not display numbers
time_slider.valtext.set_visible(False)
channel_slider.valtext.set_visible(False)

# Slider update function
def update(val):
    channel_start = int(channel_slider.val)
    time_start = int(time_slider.val)
    global current_channels
    global current_time
    current_channels = list(range(channel_start, channel_start + n_display_channels))[::-1]
    current_time = [time_start, time_start + n_display_windows*freq]
    plot_eeg()

time_slider.on_changed(update)
channel_slider.on_changed(update)

# Adjust slider direction to slide from top to bottom
ax_channel.invert_yaxis()

plt.show()