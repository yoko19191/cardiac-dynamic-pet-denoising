#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import re
import warnings

import numpy as np

import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import pydicom

from skimage.color import rgb2gray
from skimage.exposure import histogram, cumulative_distribution
from skimage import filters
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.util import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim
from brisque import BRISQUE


from tqdm import tqdm
#from tqdm.notebook import tqdm

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def timer_decorator(func):
    """
    A decorator to measure and print the execution time of a function.

    This decorator wraps a given function and prints its execution time in seconds
    after it's called. The original function's return value is passed through.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function.

    Example:
        @timer_decorator
        def my_function():
            # some code here

        my_function()  # Prints the execution time of my_function
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} running: {end_time - start_time} second")
        return result
    return wrapper


def seed_everything(seed=42):
    """Set the seed for all random number generators to ensure reproducibility.

      This function sets the seeds for multiple random number generators, including
      Python's built-in random module, NumPy, and PyTorch.

      Args:
          seed (int, optional): The seed value to set. Defaults to 0.

      Prints:
          A statement with the set seed value.

      Example:
          >>> seed_everything(seed=42)
      """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Every seed set to \"{seed}\"! ")

def display_4d_image(images, min_intensity=None, max_intensity=None):
    """
    Displays a 4D image along with its intensity distribution.

    This function provides interactive sliders to navigate through time and slices, and adjust the window level and width.
    It supports displaying images along the X, Y, or Z axis.

    Args:
        images (np.ndarray): A 4D numpy array containing the image data.
        min_intensity (float, optional): The minimum intensity value for display. Defaults to the minimum value in the images.
        max_intensity (float, optional): The maximum intensity value for display. Defaults to the maximum value in the images.

    Raises:
        ValueError: If the input is not a numpy array.

    Note:
        Requires ipywidgets for interactive.
    """
    if not isinstance(images, np.ndarray):
        raise ValueError('Input should be a numpy array')

    if min_intensity is None:
        min_intensity = np.min(images).astype(np.float32)
    if max_intensity is None:
        max_intensity = np.max(images).astype(np.float32)

    width_max = np.abs(min_intensity) + np.abs(max_intensity)

    def display_image(t, s, window_level, window_width, axis):
        # Calculate min and max intensity for the window level
        min_intensity_window = window_level - window_width // 2
        max_intensity_window = window_level + window_width // 2

        if axis == 'X':
            img = images[t, s, :, :].transpose()
        elif axis == 'Y':
            img = images[t, :, s, :].transpose()
        elif axis == 'Z':
            img = images[t, :, :, s].transpose()

        # Displaying image
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='hot', vmin=min_intensity_window, vmax=max_intensity_window)
        plt.colorbar()

        # Displaying histogram
        plt.subplot(1, 2, 2)
        img_clipped = np.clip(img, min_intensity_window,
                              max_intensity_window)  # Clip the image according to the window level and width
        plt.hist(img_clipped.ravel(), bins=256, color='gray', alpha=0.7,
                 range=(min_intensity_window, max_intensity_window))
        plt.title("Intensity Distribution")
        plt.tight_layout()
        plt.show()

    def update_slice_range(*args):
        if axis_selector.value == 'X':
            slice_slider.max = images.shape[1] - 1
        elif axis_selector.value == 'Y':
            slice_slider.max = images.shape[2] - 1
        elif axis_selector.value == 'Z':
            slice_slider.max = images.shape[3] - 1
        slice_slider.value = min(slice_slider.value, slice_slider.max)

    time_slider = widgets.IntSlider(min=0, max=images.shape[0] - 1, step=1, value=0, description='Time')
    slice_slider = widgets.IntSlider(min=0, max=images.shape[1] - 1, step=1, value=0, description='Slice')
    window_level_slider = widgets.FloatSlider(min=min_intensity, max=max_intensity,
                                              step=(max_intensity - min_intensity) / 1e5, value=max_intensity * 0.5,
                                              description='Window Level')
    window_width_slider = widgets.FloatSlider(min=1e-5, max=max_intensity - min_intensity,
                                              step=(max_intensity - min_intensity) / 1e5, value=width_max * 0.5,
                                              description='Window Width')
    axis_selector = widgets.RadioButtons(options=['X', 'Y', 'Z'], description='Axis')
    axis_selector.observe(update_slice_range, 'value')

    widget = interactive(display_image, t=time_slider, s=slice_slider, window_level=window_level_slider,
                         window_width=window_width_slider, axis=axis_selector)

    display(widget)


def display_image_in_detail(image, zoom_coordinates=(55, 15, 60, 60), title=None, axis_off=True):
    """
     Displays an image along with a zoomed-in area.

     This function takes an image and coordinates for a zoomed-in area, and then displays both the original image and
     the zoomed-in area side by side. The zoomed-in area is also marked with a red rectangle on the original image.

     Args:
         image (np.ndarray): A 2D numpy array containing the image data.
         zoom_coordinates (tuple): A tuple containing the coordinates for the zoomed-in area (x, y, width, height).
         title (str, optional): The title for the original image plot. Defaults to None.
         axis_off (bool, optional): If True, turns off the axis. Defaults to False.
     """
    x, y, width, height = zoom_coordinates
    zoom_area = image[x:x + width, y:y + height]

    # plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='hot')
    plt.title(title)
    rect = Rectangle((y, x), height, width, linewidth=1, edgecolor='r', facecolor='none') # mark
    plt.gca().add_patch(rect)
    if axis_off:
        plt.axis('off')

    # plot zoom-in area
    plt.subplot(1, 2, 2)
    plt.imshow(zoom_area, cmap='hot')
    if axis_off:
        plt.axis('off')

    plt.show()

@timer_decorator
def calculate_metrics(denoised_data, noisy_data, data_range):
    """Calculate various metrics between denoised and noisy data.

       This function computes PSNR, SSIM, MAE, and BRISQUE for the provided denoised
       and noisy data, which can be 2D, 3D, or 4D.

       Args:
           denoised_data (ndarray): The denoised image or volume data.
           noisy_data (ndarray): The original noisy image or volume data.
           data_range (float): The range of the data (e.g., 65535 for 16-bit images).

       Returns:
           tuple: Contains the PSNR, SSIM, MAE, and BRISQUE values for each volume and slice.

       Raises:
           AssertionError: If the denoised and noisy data are the same, or if they have different shapes.

       Example:
           >>> psnr_values, ssim_values, mae_values, brisque_values = calculate_metrics(denoised_image, noisy_image, 65535)
       """
    assert not np.array_equal(denoised_data, noisy_data), "Image pair must be different"
    assert denoised_data.shape == noisy_data.shape, "Images must have the same shape"

    # Expand input to 4D if necessary
    if denoised_data.ndim == 2:
        denoised_data = denoised_data.reshape(1, 1, *denoised_data.shape)
        noisy_data = noisy_data.reshape(1, 1, *noisy_data.shape)
    elif denoised_data.ndim == 3:
        denoised_data = denoised_data.reshape(1, *denoised_data.shape)
        noisy_data = noisy_data.reshape(1, *noisy_data.shape)

    volumes, slices, height, width = denoised_data.shape
    assert len(denoised_data.shape) == 4, "Images must be 4d"

    psnr_values = np.zeros((volumes, slices))
    ssim_values = np.zeros((volumes, slices))
    mae_values = np.zeros((volumes, slices))
    brisque = BRISQUE(url=False) # create brisque object
    brisque_values = np.zeros((volumes, slices))
    for vol_idx in range(volumes):
        for slice_idx in range(slices):
            denoised_image = denoised_data[vol_idx, slice_idx, :, :]
            noisy_image = noisy_data[vol_idx, slice_idx, :, :]
            # calculate PSNR
            psnr_val = cal_psnr(denoised_image, noisy_image, data_range=data_range)
            # calculate SSIM
            ssim_val = cal_ssim(denoised_image, noisy_image, data_range=data_range)
            # calculate MAE
            mae_val = np.mean(np.abs(denoised_image - noisy_image))
            # calculate BRISQUE
            fake_rgb_denoised_image = np.stack([0.2125 * denoised_image, 0.7154 * denoised_image, 0.0721 * denoised_image], axis=2)
            brisque_val = brisque.score(fake_rgb_denoised_image)
            if np.isnan(psnr_val) or np.isnan(ssim_val):
                warnings.warn("Encountered NaN value for PSNR or SSIM. Skipping frame.")
                continue

            psnr_values[vol_idx, slice_idx] = psnr_val
            ssim_values[vol_idx, slice_idx] = ssim_val
            mae_values[vol_idx, slice_idx] = mae_val
            brisque_values[vol_idx, slice_idx] = brisque_val
            
    return psnr_values, ssim_values, mae_values, brisque_values

# @timer_decorator
# def calculate_metrics(denoised_data, noisy_data, data_range):
#     assert not np.array_equal(denoised_data, noisy_data), "Image pair must be different"
#     assert denoised_data.shape == noisy_data.shape, "Images must have the same shape"

#     # Expand input to 4D if necessary
#     if denoised_data.ndim == 2:
#         denoised_data = denoised_data.reshape(1, 1, *denoised_data.shape)
#         noisy_data = noisy_data.reshape(1, 1, *noisy_data.shape)
#     elif denoised_data.ndim == 3:
#         denoised_data = denoised_data.reshape(1, *denoised_data.shape)
#         noisy_data = noisy_data.reshape(1, *noisy_data.shape)

#     volumes, slices, height, width = denoised_data.shape
#     assert len(denoised_data.shape) == 4, "Images must be 4d"

#     psnr_values = np.zeros((volumes, slices))
#     ssim_values = np.zeros((volumes, slices))
#     mae_values = np.zeros((volumes, slices))

#     for vol_idx in range(volumes):
#         for slice_idx in range(slices):
#             denoised_image = denoised_data[vol_idx, slice_idx, :, :]
#             noisy_image = noisy_data[vol_idx, slice_idx, :, :]
#             psnr_val = cal_psnr(denoised_image, noisy_image, data_range=data_range)
#             ssim_val = cal_ssim(denoised_image, noisy_image, data_range=data_range)
#             mae_val = np.mean(np.abs(denoised_image - noisy_image))

#             if np.isnan(psnr_val) or np.isnan(ssim_val):
#                 warnings.warn("Encountered NaN value for PSNR or SSIM. Skipping frame.")
#                 continue

#             psnr_values[vol_idx, slice_idx] = psnr_val
#             ssim_values[vol_idx, slice_idx] = ssim_val
#             mae_values[vol_idx, slice_idx] = mae_val

#     return psnr_values, ssim_values, mae_values




####################################################

def add_noise_to_tensor(x, noise_type, noise_level):
    """
    adding noise on torch image
    :param x:
    :param noise_type: either 'gauss' or 'poiss'
    :param noise_level:
    :return: noisy
    """
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level

    return noisy


def calc_gray_overcast(volume):
    """
    calculate statistical from a 3D numpy image
    :param volume: numpy image
    :return: dict
    """
    # Flatten the 3D volume into 1D array
    flat_pixels = volume.ravel()

    # Create a dictionary to store the results
    volume_stats = {}

    # Compute and store the statistics for the volume
    volume_stats['dtype'] = volume.dtype
    volume_stats['shape'] = volume.shape
    volume_stats['Mean'] = np.mean(flat_pixels)
    volume_stats['Std'] = np.std(flat_pixels)
    volume_stats['Min'] = np.min(flat_pixels)
    volume_stats['Median'] = np.median(flat_pixels)
    volume_stats['P_80'] = np.percentile(flat_pixels, 80)
    volume_stats['P_90'] = np.percentile(flat_pixels, 90)
    volume_stats['P_99'] = np.percentile(flat_pixels, 99)
    volume_stats['Max'] = np.max(flat_pixels)

    return volume_stats



def plot_intensity_histogram(image):
    """
    plot histogram from given image
    :param image:
    :return:
    """
    # Calculate histogram
    freq, bins = histogram(image)

    # Plot histogram
    plt.step(bins, freq*1.0/freq.sum())
    plt.xlabel('intensity value')
    plt.ylabel('fraction of pixels')
    plt.show()


def plot_cdf(image):
    """
    Plot the cumulative distribution function of an image.

    Parameters:
    image (ndarray): Input image.
    """

    # Convert the image to grayscale if needed
    if len(image.shape) == 3:
        image = rgb2gray(image[:, :, :3])

    # Compute the cumulative distribution function
    intensity = np.round(image * 255).astype(np.uint8)
    freq, bins = cumulative_distribution(intensity)

    # Plot the actual and target CDFs
    target_bins = np.arange(256)
    target_freq = np.linspace(0, 1, len(target_bins))
    plt.step(bins, freq, c='b', label='Actual CDF')
    plt.plot(target_bins, target_freq, c='r', label='Target CDF')

    # Plot an example lookup
    example_intensity = 50
    example_target = np.interp(freq[example_intensity], target_freq, target_bins)
    plt.plot([example_intensity, example_intensity, target_bins[-11], target_bins[-11]],
             [0, freq[example_intensity], freq[example_intensity], 0],
             'k--',
             label=f'Example lookup ({example_intensity} -> {example_target:.0f})')

    # Customize the plot
    plt.legend()
    plt.xlim(0, 255)
    plt.ylim(0, 1)
    plt.xlabel('Intensity Values')
    plt.ylabel('Cumulative Fraction of Pixels')
    plt.title('Cumulative Distribution Function')

    return freq, bins, target_freq, target_bins

def make_hist_equalize(image, method='cdf', **kwargs):
    # Check if the image is grayscale or RGB
    is_gray = len(image.shape) == 2

    if method == 'sigmoid':
        a = kwargs.get('a', 1)
        image_eqz = (1 + np.tanh(a * image)) / 2
    elif method == 'exponential':
        alpha = kwargs.get('alpha', 1)
        image_eqz = 1 - np.exp(-alpha * image)
    elif method == 'power':
        alpha = kwargs.get('alpha', 1)
        image_eqz = image ** alpha
    elif method == 'adaptive':
        clip_limit = kwargs.get('clip_limit', 0.03)
        tile_size = kwargs.get('tile_size', (8, 8))
        image_eqz = exposure.equalize_adapthist(
                image, clip_limit=clip_limit, nbins=256, kernel_size=(tile_size[0], tile_size[1]))
    elif method == 'gamma':
        gamma = kwargs.get('gamma', 1.0)
        image_eqz = exposure.adjust_gamma(image, gamma)
    elif method == 'contrast_stretching_percentile':
        lower_percentile = kwargs.get('lower_percentile', 5)
        upper_percentile = kwargs.get('upper_percentile', 95)
        in_range = tuple(np.percentile(image, (lower_percentile, upper_percentile)))
        image_eqz = exposure.rescale_intensity(image, in_range)
    elif method == 'unsharp_masking':
        radius = kwargs.get('radius', 5)
        amount = kwargs.get('amount', 1.0)
        blurred_image = filters.gaussian(image, sigma=radius, multichannel=not is_gray)
        image_eqz = (image + (image - blurred_image) * amount).clip(0, 1)
    elif method == 'equalize_hist_rgb':
        image_eqz = exposure.equalize_hist(image)
    elif method == 'equalize_hist_hsv':
        if not is_gray:
            hsv_image = color.rgb2hsv(image[:, :, :3])
            hsv_image[:, :, 2] = exposure.equalize_hist(hsv_image[:, :, 2])
            image_eqz = color.hsv2rgb(hsv_image)
    elif method == 'equalize_hist_yuv':
        if not is_gray:
            yuv_image = color.rgb2yuv(image[:, :, :3])
            yuv_image[:, :, 0] = exposure.equalize_hist(yuv_image[:, :, 0])
            image_eqz = color.yuv2rgb(yuv_image)
    elif method == 'cdf':
        if not is_gray:
            intensity = np.round(image * 255).astype(np.uint8)
            freq, bins = cumulative_distribution(intensity)
            target_bins = np.arange(256)
            target_freq = np.linspace(0, 1, len(target_bins))
            new_vals = np.interp(freq, target_freq, target_bins)
            image_eqz = img_as_ubyte(new_vals[img_as_ubyte(rgb2gray(image[:, :, :3]))].astype(int))
        else:
            # Handle grayscale images
            intensity = np.round(image * 255).astype(np.uint8)
            freq, bins = cumulative_distribution(intensity)
            target_bins = np.arange(256)
            target_freq = np.linspace(0, 1, len(target_bins))
            new_vals = np.interp(freq, target_freq, target_bins)
            image_eqz = img_as_ubyte(new_vals[intensity].astype(int))
    else:
        raise ValueError("Unknown method: {}".format(method))

    return image_eqz



def plot_2d_data(data_2d, title):
    """
    plot 3D wireframe from 2D data
    :param data_2d: ndarray
    :param title: string
    :return: None
    """
    print(f"argmax: {np.unravel_index(np.argmax(data_2d), data_2d.shape)}")
    print(f"argmin: {np.unravel_index(np.argmin(data_2d), data_2d.shape)}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Time')

    x = np.arange(data_2d.shape[1])
    y = np.arange(data_2d.shape[0])

    X, Y = np.meshgrid(x, y)

    # ax.plot_surface(X, Y, data_2d)
    ax.plot_wireframe(X, Y, data_2d)

    plt.title(title)

    plt.show()







