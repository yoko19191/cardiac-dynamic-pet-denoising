#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pydicom
from tqdm import tqdm
# from tqdm.notebook import tqdm


import torch 
from torch.utils.data import Dataset
import torch.nn.functional as F

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



def split_data(data_array):
    """split ndarray into train, test, vali arrays

    Args:
        data_array (numpy.ndarray): The input data array.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Train, Validation, Test arrays.
    """
    test_array = data_array[0:1]
    rest_array = data_array[1:]

    total_samples = rest_array.shape[0]
    
    train_ration = 0.8
    train_length = int(train_ration * total_samples)
    val_length = total_samples - train_length
    
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_array = rest_array[indices[:train_length]]
    val_array = rest_array[indices[train_length:]]

    return train_array, val_array, test_array

# class Mask2_5Dataset(Dataset):
#     def __init__(self, noisy_data, apply_mask=True, n_mask = 1):
#         self.noisy_data = noisy_data
#         self.apply_mask = apply_mask
#         self.n_mask = n_mask

#     def __len__(self):
#         return self.noisy_data.shape[0] * (self.noisy_data.shape[2] - 2)

#     def __getitem__(self, idx):
#         time_idx = idx // (self.noisy_data.shape[2] - 2)
#         depth_idx = idx % (self.noisy_data.shape[2] - 2)
#         x_top = self.noisy_data[time_idx, :, depth_idx]
#         x_middle = self.noisy_data[time_idx, :, depth_idx + 1]
#         x_bottom = self.noisy_data[time_idx, :, depth_idx + 2]

#         if self.apply_mask:
#             x_middle, mask_middle = self.mask(x_middle)
#             return x_top, x_middle, x_bottom, mask_middle
#         else:
#             return x_top, x_middle, x_bottom

#     def mask(self, x):
#         n = self.n_mask  # Number of pixels to mask
#         mask_middle = torch.ones_like(x)  # Initialize the mask
#         x_noised = x.clone()  # Create a copy for the noised data
#         random_idx = torch.randint(0, x.numel(), (n,))  # Random indexing for tensors
#         mask_middle.view(-1)[random_idx] = 0
#         #x_noised.view(-1)[random_idx] = torch.normal(mean=0, std=1, size=(n,))  # or any noise you want to introduce
#         x_noised.view(-1)[random_idx] = 0
#         return x_noised, mask_middle

# class Mask2_5Dataset(Dataset):
#     """A dataset class for blind-spot network.

#     Args:
#         noisy_tensor (Tensor): Input tensor of shape (batch, time, channel, depth, height, width).
#         apply_mask (bool): Whether to apply the mask.
#         n_mask (int): Number of pixels to mask.
#    """
#     def __init__(self, noisy_tensor, apply_mask=True, n_mask=1):
#         self.noisy_tensor = noisy_tensor
#         self.apply_mask = apply_mask
#         self.n_mask = n_mask

#     def __len__(self):
#         # batch shape : (batch * time * (three successive depth_slices))
#         return self.noisy_tensor.shape[0] * self.noisy_tensor.shape[1] * (self.noisy_tensor.shape[3] - 2)
    
#     def __getitem__(self, idx):
#         batch_idx = idx // (self.noisy_tensor.shape[1] * (self.noisy_tensor.shape[3] - 2))
#         time_idx = (idx // (self.noisy_tensor.shape[3] - 2)) % self.noisy_tensor.shape[1]
#         depth_idx = idx % (self.noisy_tensor.shape[3] - 2)

#         x_top = self.noisy_tensor[batch_idx, time_idx, :, depth_idx, :, :]
#         x_middle = self.noisy_tensor[batch_idx, time_idx, :, depth_idx + 1, :, :]
#         x_bottom = self.noisy_tensor[batch_idx, time_idx, :, depth_idx + 2, :, :]

#         if self.apply_mask:
#             x_middle, mask_middle = self.mask(x_middle)
#             return x_top, x_middle, x_bottom, mask_middle
#         else:
#             return x_top, x_middle, x_bottom

#     def mask(self, x):
#         n = self.n_mask  # Number of pixels to mask
#         mask_middle = torch.ones_like(x)  # Initialize the mask
#         x_noised = x.clone()  # Create a copy for the noised data
#         random_idx = torch.randint(0, x.numel(), (n,))  # Random indexing for tensors
#         mask_middle.view(-1)[random_idx] = 0
#         #x_noised.view(-1)[random_idx] = torch.normal(mean=0, std=1, size=(n,))  # or any noise you want to introduce
#         x_noised.view(-1)[random_idx] = 0
#         return x_noised, mask_middle


def restore_data(normalized_data, restore_info):
    
    original_min = restore_info["original_min"]
    original_max = restore_info["original_max"]
    z_score_mean = restore_info["z_score_mean"]
    z_score_std_dev = restore_info["z_score_std_dev"]
    noise_min = restore_info["noise_min"]
    noise_max = restore_info["noise_max"]

    denormalized_data = normalized_data * (noise_max - noise_min) + noise_min

    # un z-score
    un_z_scored_data = denormalized_data * z_score_std_dev + z_score_mean

    # restore range 
    restored_data = np.clip(un_z_scored_data, original_min, original_max).astype(np.int16)

    return restored_data


class MaskDataset(Dataset):
    """ 
    Dataset for Blind-spot network
    Args:
    - data_tensor : the input tensor with dimensions(patient, time, channel, depth, height, width)
    - num_mask : the number of pixel to mask middle slice. Default is 1
    """
    def __init__(self, data_tensor, num_mask=1):
        self.data_tensor = data_tensor
        self.num_mask = num_mask
        assert len(data_tensor.shape) == 6, "noisy_tensor should have 6 dimensions (patient, time, channel, depth, height, width)"
        assert self.data_tensor.size(3) >= 3, "Depth should be at least 3 for 2.5D slices."
        
        self.p = data_tensor.shape[0] # number of patiences
        self.t = data_tensor.shape[1] # number of time frame
        self.d = data_tensor.shape[3] - 2 # number of contines slices
        
    def __len__(self):
        return self.p * self.t * self.d # number of continues three slices
    
    def __getitem__(self, idx):
        patience_idx = idx // (self.t * self.d)
        time_idx = (idx % (self.t * self.d)) // self.d
        depth_idx = idx % self.d + 1  # We add 1 to start from the second slice (for the middle slice)

        # Extract the slices
        top_slice = self.data_tensor[patience_idx, time_idx, :, depth_idx-1, :, :]
        middle_slice = self.data_tensor[patience_idx, time_idx, :, depth_idx, :, :].clone()  # We use clone() to avoid in-place modifications
        bottom_slice = self.data_tensor[patience_idx, time_idx, :, depth_idx+1, :, :]
        
        # create mask position(s)
        mask_middle = middle_slice.clone()
        H, W = middle_slice.shape[-2], middle_slice.shape[-1]
        device = middle_slice.device
        random_h_indices = torch.randint(high=H, size=(self.num_mask,)).to(device)
        random_w_indices = torch.randint(high=W, size=(self.num_mask,)).to(device)

        # Generate gaussian noise as mask
#         noise = torch.randn_like(mask_middle).to(device)
#         for i in range(self.num_mask):
#             mask_middle[:, random_h_indices[i], random_w_indices[i]] += noise[:, random_h_indices[i], random_w_indices[i]]
            
#         # Clip the values to ensure they remain in the original range
#         min_val, max_val = torch.min(middle_slice), torch.max(middle_slice)
#         mask_middle = torch.clamp(mask_middle, min_val, max_val)
        
        #
        min_val, max_val = torch.min(middle_slice), torch.max(middle_slice)
        noise_range = 1.00 * (max_val - min_val)
        for i in range(self.num_mask):
            # Generate a single noise value for each channel
            #noise_value = torch.randn(size=(mask_middle.shape[0],)).to(device) * noise_range
            #noise_value = torch.clamp(noise_value, -noise_range, noise_range)  # Ensure the noise is within the range
            noise_value = max_val
            # Add the noise to the specific location in mask_middle
            mask_middle[:, random_h_indices[i], random_w_indices[i]] += noise_value

        
        
        return top_slice, mask_middle, bottom_slice, middle_slice
    
        
class NACDataset(Dataset):
    def __init__(self, data_tensor, mode='train', noise_type='gaussian'):
        self.data_tensor = data_tensor
        self.mode = mode
        self.noise_type = noise_type  # Can be 'gaussian' or 'poisson'
        assert len(data_tensor.shape) == 6, "noisy_tensor should have 6 dimensions (patient, time, channel, depth, height, width)"
        assert self.data_tensor.size(3) >= 3, "Depth should be at least 3 for 2.5D slices."
        
        self.p = data_tensor.shape[0]  # number of patient
        self.t = data_tensor.shape[1]  # number of time frame
        self.d = data_tensor.shape[3] - 2  # number of continuous slices

    def mad_estimate(self, image_tensor):
        median = torch.median(image_tensor)
        mad = torch.median(torch.abs(image_tensor - median))
        noise_level = mad / 0.6745
        return torch.mean(noise_level)

    # def mae_estimate(self, image_tensor):
    #     lambda_estimate = torch.mean(image_tensor)
    #     std_estimate = torch.sqrt(lambda_estimate)
    #     return std_estimate
    
    def anscombe_estimate(self, image_tensor, window_size=7):
        def anscombe(x):
            return 2 * torch.sqrt(x + 3 / 8)

        def inv_anscombe(y):
            return (y / 2) ** 2 - 3 / 8
        
        # Reshape tensor to match expected input shape for F.avg_pool2d
        c, h, w = image_tensor.shape
        reshaped_tensor = image_tensor.view(c, 1, h, w)
        
        # Calculate local mean using average pooling
        local_mean = F.avg_pool2d(reshaped_tensor, window_size, stride=1, padding=window_size // 2)
        
        # Calculate local variance
        local_var = F.avg_pool2d(reshaped_tensor ** 2, window_size, stride=1, padding=window_size // 2) - local_mean ** 2
        
        # Calculate average std
        noise_std = torch.sqrt(F.relu(local_var)).mean()
        
        return noise_std

    def __len__(self):
        return self.p * self.t * self.d  # number of continuous three slices

    def __getitem__(self, idx):
        patient_idx = idx // (self.t * self.d)
        time_idx = (idx % (self.t * self.d)) // self.d
        depth_idx = idx % self.d + 1  # We add 1 to start from the second slice (for the middle slice)

        # Extract the slices
        top_slice = self.data_tensor[patient_idx, time_idx, :, depth_idx-1, :, :]
        middle_slice = self.data_tensor[patient_idx, time_idx, :, depth_idx, :, :]
        bottom_slice = self.data_tensor[patient_idx, time_idx, :, depth_idx+1, :, :]
        middle_target = self.data_tensor[patient_idx, time_idx, :, depth_idx, :, :].clone()  # noisy middle slice as clean

        if self.mode == 'train':
            # Concatenate slices to estimate noise
            concatenated_slices = torch.cat((top_slice, middle_slice, bottom_slice), dim=0)
            #noise_shape = middle_slice.shape
            # Estimate noise std based on noise_type
            if self.noise_type == 'gaussian':
                estimate_std = self.mad_estimate(concatenated_slices)
                simulated_noise = torch.normal(mean=0., std=estimate_std, size=middle_slice.shape).to(middle_slice.device)  # Move noise to the same device
            elif self.noise_type == 'poisson':
                # estimate_std = self.mae_estimate(concatenated_slices)
                estimate_std = self.anscombe_estimate(concatenated_slices)
                simulated_noise = torch.poisson(estimate_std * torch.ones_like(middle_slice)).to(middle_slice.device) - estimate_std
                simulated_noise = simulated_noise * 0.1
            else:
                raise ValueError("Invalid noise_type. Choose either 'gaussian' or 'poisson'.")
            
            # Add simulated noise
            top_slice += simulated_noise
            middle_slice += simulated_noise
            bottom_slice += simulated_noise

        return top_slice, middle_slice, bottom_slice, middle_target
    
    
        
class Nb2Nb2D_Dataset(Dataset):
    def __init__(self, data_tensor, k=2):
        """
        Args:
            data (torch.Tensor): A tensor of shape [patient, time, channel, depth, height, width].
            k (int): Size of the cell for sub-sampling in random_neighbor_subsample.
        """
        assert len(data_tensor.shape) == 6, "The input data should have 6 dimensions [patient, time, channel, depth, height, width]"
        
        self.data_tensor = data_tensor
        self.k = k
        self.p = data_tensor.shape[0]  # number of patient
        self.t = data_tensor.shape[1]  # number of time frame
        self.d = data_tensor.shape[3]  # number of continuous slices

        # Pre-compute the total number of slices we have across all patients and times
        self.num_slices = data_tensor.shape[0] * data_tensor.shape[1] * data_tensor.shape[3]

    def __len__(self):
        return self.num_slices

    
    # def random_neighbor_subsample(self, tensor, k):
    #     """
    #     Perform random neighbor sub-sampling on a tensor of shape CxHxW.
        
    #     Args:
    #     - tensor (torch.Tensor): Input tensor of shape CxHxW.
    #     - k (int): Size of the cell for sub-sampling.
        
    #     Returns:
    #     - g1, g2 (torch.Tensor, torch.Tensor): Two sub-sampled tensors, each of shape Cx(H//k)x(W//k).
    #     """
    #     # Check tensor dimensions
    #     C, H, W = tensor.shape
        
    #     # Initialize sub-sampled tensors
    #     g1 = torch.zeros((C, H//k, W//k))
    #     g2 = torch.zeros((C, H//k, W//k))
        
    #     for i in range(0, H, k):
    #         for j in range(0, W, k):
    #             # Randomly select one of the kxk neighbors for g1 and another for g2
    #             neighbors = [(i+x, j+y) for x in range(k) for y in range(k)]
    #             idx1, idx2 = torch.randperm(k*k)[:2]  # Randomly select two indices
    #             g1[:, i//k, j//k] = tensor[:, neighbors[idx1][0], neighbors[idx1][1]]
    #             g2[:, i//k, j//k] = tensor[:, neighbors[idx2][0], neighbors[idx2][1]]
        
    #     return g1, g2
    
    def random_neighbor_subsample(self, tensor, k):
        """
        Perform random neighbor sub-sampling on a tensor of shape CxHxW.
        
        Args:
        - tensor (torch.Tensor): Input tensor of shape CxHxW.
        - k (int): Size of the cell for sub-sampling.
        
        Returns:
        - g1, g2 (torch.Tensor, torch.Tensor): Two sub-sampled tensors, each of shape Cx(H//k)x(W//k).
        """
        C, H, W = tensor.shape
        unfolded = tensor.unfold(1, k, k).unfold(2, k, k)
        unfolded = unfolded.contiguous().view(C, H//k, W//k, k*k)
        
        idx1, idx2 = torch.randperm(k*k)[:2].to(tensor.device)
        
        g1 = unfolded[..., idx1].squeeze(-1)
        g2 = unfolded[..., idx2].squeeze(-1)
        
        return g1, g2
    
    def __getitem__(self, idx):
        # Given an idx, find out which patient, time, and depth this corresponds to
        depth_idx = idx % self.data_tensor.shape[3]
        time_idx = (idx // self.data_tensor.shape[3]) % self.data_tensor.shape[1]
        patient_idx = idx // (self.data_tensor.shape[3] * self.data_tensor.shape[1])
        
        # Extract the slice
        slice_2d = self.data_tensor[patient_idx, time_idx, :, depth_idx, :, :]
        
        # Get the sub-sampled tensors g1 and g2
        g1, g2 = self.random_neighbor_subsample(slice_2d, self.k)

        return slice_2d, g1, g2

    

    
