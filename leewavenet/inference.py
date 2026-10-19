import numpy as np
import torch


def segmentation(segmentation_model, da):
    """Function to perform segmentation using the segmentation model"""
    wavemask = segmentation_model.predict(da.values)[0].numpy()
    return wavemask


def amplitude(amplitude_model, da):
    """Function to predict amplitude using the amplitude model"""
    amp = amplitude_model.predict(torch.Tensor(np.array([da, da, da])))[0][0]
    return amp.numpy()


def wavelength(wavelength_model, da):
    """Function to predict wavelength using the wavelength model"""
    wl = wavelength_model.predict(torch.Tensor(np.array([da, da, da])))[0][0]
    return wl.numpy()


def orientation(orientation_model, da):
    """Function to predict orientation using the orientation model"""
    orient = orientation_model.predict(torch.Tensor(np.array([da, da, da])))[0]
    orient_deg = 180 * np.arctan(orient[0] / orient[1]) / np.pi
    return orient_deg
