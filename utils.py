import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def generate_muap(
        pars : np.ndarray,
        generator,
        reps : int = 10,
        return_all : bool = False):
    """
    Generates muaps from a given parameter set

    Args
    ----
    pars : np.ndarray, [n_MU, 6]
        order of pars is fixed* : [fd, d, a, iz, cv, fl]
    generator : Biomime.models.generator.Generator object
    reps : number of repetitions of generation (z values, see Biomime)
    return_all : returns all repetitions generated

    """
    if pars.ndim == 1:
        pars = pars[None, :]

    n_MU = pars.shape[0]
    pars = torch.from_numpy(pars)
    sim_muaps = []
    
    # Ensure device compatibility
    device = next(generator.parameters()).device

    for _ in range(reps):
        cond = pars.to(device)
        sim = generator.sample(n_MU, cond.float(), cond.device)

        sim = sim.to("cpu")
        if n_MU == 1:
            sim = sim.permute(1, 2, 0).detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()
        sim_muaps.append(sim)

    muap = np.array(sim_muaps).mean(0)

    if return_all:
        return np.array(sim_muaps)

    return muap

def add_to_path(folder_list):
    for folder_path in folder_list:
        # Recursively add all subdirectories to the Python path
        for root, dirs, files in os.walk(folder_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                sys.path.append(dir_path)

    return None


def plot_muap_simple(muap, scale=True, overlap=False, snap=False):
    """
    Simple plot util for MUAPs

    Args:
    -----
        scale : bool, normalizes all channels before plotting
            use-case ?
        overlap : bool, whether rows of the MUAP should overlap
        snap : bool, get a snapshot of the MUAP
    
    """
    # Scaling
    if scale:
        muap_max = np.nanmax(muap)
        muap_min = np.nanmin(muap)
        muap = (muap - muap_min) / (muap_max - muap_min)

    n_row, n_col, n_time = muap.shape

    # Reshaping for being plottable
    mynans = np.zeros((n_row, n_col, int(n_time/2))) * np.nan
    plotable = np.dstack((muap, mynans))
    plotable = plotable.reshape([n_row, -1])

    if snap:
        plt.figure(figsize=[3, 3])
        plotable -= np.linspace(0, n_row-1, n_row)[:, None] * 0.35
        plt.plot(plotable.T, linewidth=0.5)
        plt.axis('off')

        return None

    # For spacing channels out
    if overlap:
        spacing = 0.5
    else:
        spacing = 1

    plotable -= np.linspace(0, n_row-1, n_row)[:, None] * spacing
    
    plt.figure(figsize=[int(n_col*0.6), 6])
    plt.plot(plotable.T, linewidth=0.5)
    # plt.ylabel() : add electrode rows and columns numbers
    plt.axis('off')

    # TO-DO: add simple time-axis with text box
    # TO-DO: add max V on the y-axis

    return None


def plot_linear_array(data, highlight=None, scale=True, overlap=True, axis=None):
    """
    Plots a linear "column" of array, assumed to be oriented to fiber direction
    """
    # Scaling
    if scale:
        data_max = np.nanmax(data)
        data_min = np.nanmin(data)
        data = (data - data_min) / (data_max - data_min)

    n_row, n_time = data.shape

    # For spacing channels out
    if overlap:
        spacing = 0.25
    else:
        spacing = 1

    data -= np.linspace(0, n_row-1, n_row)[:, None] * spacing

    if axis is None:
        plt.figure(figsize=[3, 4])
        axis = plt.gca()

    axis.plot(data.T, linewidth=0.5)

    if highlight is not None:
        mask = np.ones_like(data) * np.nan
        mask[highlight] = 1
        data *= mask
        axis.plot(data.T, linewidth=1.5)

    axis.axis('off')
    # TO-DO: add simple time-axis with text box
    # TO-DO: add max V on the y-axis
    # TO-DO: maybe add electrode numbers on the y-axis

    return None


def get_nrmse(x, y, type='range'):
    """
    Compute normalized root mean squared difference between x and y.

    Args:
    ----
        x, y : np.ndarray, same shape
        type : 'range' - divide by range of the first observation
               'power' - divide by average power of the two
    """

    rmse = np.sqrt(np.mean((x-y)**2))
    if type == 'range':
        range = np.max(x) - np.min(x)
        rmse = rmse/range
    elif type == 'power':
        pow1 = np.linalg.norm(x)
        pow2 = np.linalg.norm(y)
        rmse = rmse/(np.mean((pow1, pow2)))

    return rmse


def filter_muaps(muaps):
    # low-pass filtering for smoothing
    fs = 4096.
    ndim = muaps.ndim

    if ndim == 4:
        n_MU, n_row, n_col, time_samples = muaps.shape

    T = time_samples / fs
    cutoff = 800
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    if ndim == 4:
        filtered_muaps = filtfilt(b, a, muaps.reshape(-1, time_samples))
        filtered_muaps = filtered_muaps.reshape(n_MU, n_row, n_col, time_samples)

    return filtered_muaps
