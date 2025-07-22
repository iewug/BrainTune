'''
Calculate DE feature distance
Verify that distance within the same person < distance for different experiments of the same person < distance between different people
'''
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import seaborn as sns
# sns.set_style("whitegrid")
from matplotlib import rcParams
# Set global font to Times New Roman
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'medium'  # Adjust font weight
})
rcParams['font.size'] = 10

################
# get features #
################
print('Loading Data...')
nameList = [f'{i}_{j}' for i in range(1,16) for j in range(1,4)]
nameList.remove('4_1') # This subject deviates significantly from the others.
dir_name = '../data/seed3/'
DEs_within = []
DEs_cross = []
for name in nameList:
    featureList = []
    featureList_cross = []
    f = h5py.File(f'{dir_name}/{name}/de.h5')
    de = f[name]['de'][:] # (len, 5, ch_num)
    trialStart = f[name].attrs['trialStart']
    trialEnd = f[name].attrs['trialEnd']
    # The mask seems to avoid some (extremely few) excessively large Euclidean distances.
    mask = np.load(f'{dir_name}/{name}/artifact/all.npy').T # (sample_num_all, ch_num)
    for i in range(15): # Extract n samples from each video.
        videoDE = de[trialStart[i]:trialEnd[i]]
        videoMask = mask[trialStart[i]:trialEnd[i]]
        videoMask = np.sum(videoMask,axis=1)
        goodSample = videoMask <= 15
        videoDE = videoDE[goodSample]
        # Randomly select n indices. Sampling is needed, otherwise it will be slow and cause memory overflow.
        indices = np.random.choice(videoDE.shape[0], 9, replace=False)
        indices_cross = np.random.choice(indices, 2, replace=False)
        videoDE_within = videoDE[indices]
        videoDE_cross = videoDE[indices_cross]
        featureList.append(videoDE_within)
        featureList_cross.append(videoDE_cross)
    featureList = np.concatenate(featureList)
    featureList_cross = np.concatenate(featureList_cross)
    featureList = featureList.reshape(featureList.shape[0],-1) # (sample_num, feature_dim)
    featureList_cross = featureList_cross.reshape(featureList_cross.shape[0],-1) # (sample_num, feature_dim)
    DEs_within.append(featureList)
    DEs_cross.append(featureList_cross)
print('Done!')


# Classify data by subject
subject_data_within = defaultdict(list) # {'1': [three ndarrays], '2': ...}
for name, data in zip(nameList, DEs_within):
    subject_id = name.split('_')[0]  # Extract subject ID, e.g., '1', '2'
    subject_data_within[subject_id].append(data)

subject_data_cross = defaultdict(list) # {'1': [three ndarrays], '2': ...}
for name, data in zip(nameList, DEs_cross):
    subject_id = name.split('_')[0]  # Extract subject ID, e.g., '1', '2'
    subject_data_cross[subject_id].append(data)


# Calculate distance within the same subject, same experiment
print('Calculating distance within the same subject, same experiment...')
single_subject_distances = []
for data in DEs_within:
    # data is a (sample_num, feature_dim) ndarray
    # Calculate the distance matrix between every two samples in data, and extract the upper triangle (to avoid redundant calculations).
    dist_matrix = cdist(data, data, metric='euclidean')
    # Extract the upper triangle (excluding the diagonal), which are the Euclidean distances between samples.
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    single_subject_distances.extend(dist_matrix[triu_indices])
print(len(single_subject_distances))
print('Done!')
single_subject_distances = [x for x in single_subject_distances if x <= 80]


width = 4
height = 3.5 # Height, maintain 4:3 aspect ratio

# Set figsize
plt.figure(figsize=(width, height))
# plt.hist(single_subject_distances, bins=50, edgecolor='black', color='red', alpha=0.4,label='same subject; same exp')
sns.kdeplot(single_subject_distances, label='Same Sub;\nSame Exp', fill=True, alpha=0.4, linewidth=1)

# Calculate distance between different experiments of the same subject
print('Calculating offset between different experiments of the same subject...')
within_subject_distances = []
for data_list in subject_data_within.values():
    # data_list is a list containing data from different experiments of the same person, each element is a (sample_num, feature_dim) ndarray.
    exp_num = len(data_list) # Number of experiments
    for i in range(exp_num):
        for j in range(i + 1, exp_num):
            dist = cdist(data_list[i], data_list[j], metric='euclidean') # (sample_num, sample_num)
            within_subject_distances.extend(dist.flatten())
print(len(within_subject_distances))
print('Done!')
within_subject_distances = [x for x in within_subject_distances if x <= 80]

# Plot histogram
# plt.hist(random.sample(within_subject_distances, 397980), bins=50, edgecolor='black', color='green',alpha=0.4,label='same subject; dif exp')
sns.kdeplot(within_subject_distances, label='Same Sub;\nDif Exp', fill=True, alpha=0.4, linewidth=1)


# Calculate offset between different subjects
print('Calculating offset between different subjects...')
between_subject_distances = []
subject_ids = list(subject_data_cross.keys()) # ['1', '2', ...]
for i in range(len(subject_ids)):
    for j in range(i + 1, len(subject_ids)):
        # Get data lists from two different subjects
        data_list_i = subject_data_cross[subject_ids[i]] # 3 (sample_num, feature_dim) ndarrays
        data_list_j = subject_data_cross[subject_ids[j]]
        # Calculate pairwise distances between all experiment samples for the two subjects
        for sample_i in data_list_i:
            for sample_j in data_list_j:
                dist = cdist(sample_i, sample_j, metric='euclidean')
                between_subject_distances.extend(dist.flatten())
print(len(between_subject_distances))
print('Done!')
between_subject_distances = [x for x in between_subject_distances if x <= 80]

# Plot histogram
# plt.hist(random.sample(between_subject_distances, 397980), bins=50, edgecolor='black', color='blue',alpha=0.4,label='dif subject')
sns.kdeplot(between_subject_distances, label='Dif Sub', fill=True, alpha=0.4, linewidth=1)


# plt.legend()
# plt.xlabel("Euclidean Distance")
# plt.ylabel("Density")
plt.tight_layout()
plt.savefig('out.pdf')
exit()

# Convert to numpy arrays for calculation
within_subject_distances = np.array(within_subject_distances)
between_subject_distances = np.array(between_subject_distances)

# Calculate mean and standard deviation
within_mean = np.mean(within_subject_distances)
between_mean = np.mean(between_subject_distances)
within_std = np.std(within_subject_distances)
between_std = np.std(between_subject_distances)

# Print mean and standard deviation
print(f"Average offset for the same person across multiple experiments: {within_mean:.2f} ± {within_std:.2f}")
print(f"Average offset between different people: {between_mean:.2f} ± {between_std:.2f}")

# Perform t-test to verify if the difference between the two groups of offsets is significant
t_stat, p_value = ttest_ind(within_subject_distances, between_subject_distances)
print(f"T-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")

# # Check p-value to verify significance
# if p_value < 0.05:
#     print("Results are significant: Offset between different people is greater than offset for the same person across multiple experiments.")
# else:
#     print("Results are not significant: Insufficient evidence to show that offset between different people is greater than offset for the same person across multiple experiments.")

# Check p-value and t-statistic to verify significance and direction
if p_value < 0.05:
    if t_stat > 0:
        print("Results are significant: Offset for the same person across multiple experiments is greater than offset between different people.")
    else:
        print("Results are significant: Offset between different people is greater than offset for the same person across multiple experiments.")
else:
    print("Results are not significant: Insufficient evidence to show a significant difference between offsets for the same person across multiple experiments and offsets between different people.")