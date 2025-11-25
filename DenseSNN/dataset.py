import torch
# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import os
import glob
import zipfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader





def augment(event):  
    x_shift = 4
    y_shift = 4
    theta = 10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event



def readNpSpikes(filename, max_timeStamps):
    npEvent = np.load(filename)

    # print("npy shape:", npEvent.shape)
    # print("np max :",npEvent[:, 0].max(), npEvent[:, 1].max(), npEvent[:, 2].max(), npEvent[:, 3].max())

    if npEvent.size == 0:
        return None
    else:
        # Extract timestamps from the 4th column (npEvent[:, 3])
        timestamps = npEvent[:, 3]
        startTime = timestamps.min()

        # Normalize timestamps to start at 0
        if startTime != 0:
            timestamps -= startTime

        # Convert timestamps to milliseconds
        timestamps = (timestamps * 1000).astype(int)

        # Update the npEvent array with normalized timestamps
        npEvent[:, 3] = timestamps

        # Filter to keep events within the maximum time range
        max_time = max_timeStamps  # in milliseconds
        npEvent = npEvent[timestamps <= max_time]

        if npEvent.size == 0:
            return None

        # Return the slayer.io.Event with x, y, polarity (p), and normalized timestamps (t)
        return slayer.io.Event(npEvent[:, 0], npEvent[:, 1], npEvent[:, 2], npEvent[:, 3])





class EyeDataset(Dataset):
    def __init__(
        self, path='data path', train=True, sampling_time=1, sample_length=33, transform=augment, max_timeStamps=33
    ):
        super(EyeDataset, self).__init__()
        self.path = path
        if train:
            data_path = os.path.join(path, 'train')
        else:
            data_path = os.path.join(path, 'test')

        # Use glob to find .npy files in subfolders 0 and 1
        self.samples = glob.glob(os.path.join(data_path, '0', '*.npy')) + \
                       glob.glob(os.path.join(data_path, '1', '*.npy'))

        if len(self.samples) == 0:
            raise ValueError(f"No .npy files found in {data_path}. Please check the directory and file extensions.")

        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length / sampling_time)
        self.transform = transform
        self.max_timeStamps = max_timeStamps  # Save max_timeStamps for later use

    def __getitem__(self, i):
        filename = self.samples[i]
        label = int(filename.split('/')[-2])  # Get label from the folder name (0 or 1)
        event = readNpSpikes(filename, max_timeStamps=self.max_timeStamps)  # Pass max_timeStamps here
        if self.transform is not None:
            event = self.transform(event)
        spike = event.fill_tensor(
            torch.zeros(2, 260, 360, self.num_time_bins),
            sampling_time=self.sampling_time,
        )
        # return spike, label
        return spike.reshape(-1, self.num_time_bins), label

    def __len__(self):
        return len(self.samples)
