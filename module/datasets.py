#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pydicom
from tqdm import tqdm
# from tqdm.notebook import tqdm


import torch 
from torch.utils.data import Dataset

def load_4d_dicom(patient_path):
    """
        Loads 4D DICOM images from the specified patient directory.

        This function reads all DICOM files in the given directory, validates necessary metadata, and organizes the images
        into a 4D NumPy array. The array is sorted by acquisition time, instance number, and image position.

        Args:
            patient_path (str): The path to the directory containing the patient's DICOM files.

        Raises:
            ValueError: If no DICOM files are found in the specified directory.
            ValueError: If 'InstanceNumber' is missing from any DICOM file.
            ValueError: If 'AcquisitionTime' is missing from any DICOM file.
            ValueError: If 'ImagePositionPatient' is missing from any DICOM file.

        Returns:
            np.ndarray: A 4D NumPy array containing the sorted DICOM images.
    """
    dicom_files = [pydicom.dcmread(os.path.join(patient_path, f)) for f in os.listdir(patient_path) if f.endswith('.dcm')]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in the specified directory: {patient_path}")

    # metadata = []
    for file in dicom_files:
        if 'InstanceNumber' not in file:
            raise ValueError(f"InstanceNumber is missing from DICOM file: {file.filename}")
        if 'AcquisitionTime' not in file:
            raise ValueError(f"AcquisitionTime is missing from DICOM file: {file.filename}")
        if 'ImagePositionPatient' not in file:
            raise ValueError(f"ImagePositionPatient is missing from DICOM file: {file.filename}")
        # metadata.append(file)

    dicom_files.sort(
        key=lambda x: (float(x.AcquisitionTime), int(x.InstanceNumber), [float(i) for i in x.ImagePositionPatient]))

    time_points = sorted(set(file.AcquisitionTime for file in dicom_files))

    images = []

    for time in time_points:
        time_slice_files = [file for file in dicom_files if file.AcquisitionTime == time]
        time_slices = []

        for file in time_slice_files:
            time_slices.append(file.pixel_array)

        images.append(time_slices)

    images = np.array(images)

    return images


def save_4d_dicom(dicom_folder, ndarray_4d, output_folder):
    """
        Saves 4D DICOM images to the specified output folder.

        This function reads all DICOM files from the specified folder, sorts them by acquisition time, instance number, and
        image position, and then replaces the pixel data with the data from a 4D ndarray. The new DICOM files are saved in the
        specified output folder.

        Args:
            dicom_folder (str): The path to the directory containing the original DICOM files.
            ndarray_4d (ndarray): A 4D ndarray containing the image data to be saved.
            output_folder (str): The path to the directory where the new DICOM files will be saved.

        Note:
            The output folder will be created if it does not exist.
            The original metadata of the DICOM files will be preserved, only the pixel data will be updated.
    """
    dicom_files = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

    dicom_files.sort(
        key=lambda x: (float(x.AcquisitionTime), int(x.InstanceNumber), [float(i) for i in x.ImagePositionPatient]))

    os.makedirs(output_folder, exist_ok=True)
    time_points = sorted(set(file.AcquisitionTime for file in dicom_files))

    idx = 0
    total_files = len(dicom_files)

    for t, time in enumerate(tqdm(time_points, desc="Processing time points")):
        time_slice_files = [file for file in dicom_files if file.AcquisitionTime == time]

        for d, file in enumerate(time_slice_files):
            # create new dicom file from original one
            new_dicom = pydicom.dcmread(file.filename) # makesure all meta information the same
            # only update pixel data
            new_dicom.PixelData = ndarray_4d[t, d].tobytes()
            original_filename = os.path.basename(file.filename)
            new_file_path = os.path.join(output_folder, original_filename)
            new_dicom.save_as(new_file_path)
            idx += 1

    print(f"Saved {total_files} denoised DICOM files.")


class Mask2_5Dataset(Dataset):
    def __init__(self, noisy_data, apply_mask=True, n_mask = 1):
        self.noisy_data = noisy_data
        self.apply_mask = apply_mask
        self.n_mask = n_mask

    def __len__(self):
        return self.noisy_data.shape[0] * (self.noisy_data.shape[2] - 2)

    def __getitem__(self, idx):
        time_idx = idx // (self.noisy_data.shape[2] - 2)
        depth_idx = idx % (self.noisy_data.shape[2] - 2)
        x_top = self.noisy_data[time_idx, :, depth_idx]
        x_middle = self.noisy_data[time_idx, :, depth_idx + 1]
        x_bottom = self.noisy_data[time_idx, :, depth_idx + 2]

        if self.apply_mask:
            x_middle, mask_middle = self.mask(x_middle)
            return x_top, x_middle, x_bottom, mask_middle
        else:
            return x_top, x_middle, x_bottom

    def mask(self, x):
        n = self.n_mask  # Number of pixels to mask
        mask_middle = torch.ones_like(x)  # Initialize the mask
        x_noised = x.clone()  # Create a copy for the noised data
        random_idx = torch.randint(0, x.numel(), (n,))  # Random indexing for tensors
        mask_middle.view(-1)[random_idx] = 0
        #x_noised.view(-1)[random_idx] = torch.normal(mean=0, std=1, size=(n,))  # or any noise you want to introduce
        x_noised.view(-1)[random_idx] = 0
        return x_noised, mask_middle
    
    