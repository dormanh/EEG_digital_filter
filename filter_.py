#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.io
from scipy import signal


def filt_matlab_data(
    input_path,
    output_path,
    num_of_channels,
    sampl_freq,
    low_freq,
    high_freq,
    ch_select=None,
):
    """Applies a bidirectional digital bandpass filter to an input EEG signal with an arbitrary number of channels.

    Parameters
    ----------
    input_path: str
        Path to the file containing the input data.
    output_path: str
        Path to the output file (.mat is automatically appended to the filename).
    num_of_channels: int
        Number of input channels.
    sampl_freq: int
        Sampling frequency in Hz.
    low_freq: int
        Lower bound of the bandpass filter in Hz.
    high_freq: int
        Upper bound of the bandpass filter in Hz.
    ch_select: str
        With length equal to num_of_channels, containing only '0'-s and '1'-s that
        indicate, whether the respective channel should be kept or discarded from the data. If
        len(ch_select) < num_of_channels, the missing characters are assumed to be identical
        to the last specified character.

    Returns
    ----------
        out: array_like
            Filtered signal.
    """

    # import data
    matfile = scipy.io.loadmat(input_path)
    keys = [
        k
        for k in matfile.keys()
        if k not in ["__header__", "__version__", "__globals__"]
    ]
    for k in keys:
        if type(matfile.get(k)) == np.ndarray:
            data = matfile.get(k)[0]
            break

    # in case no data in the required format can be found under any of the keys
    try:
        data
    except NameError:
        raise ValueError("Input format is unknown. Expected np.ndarray.")

    # design filter
    sos = signal.butter(
        10,
        (low_freq, high_freq),
        btype="bandpass",
        fs=sampl_freq,
        output="sos",
    )

    # fill missing characters in ch_select if needed
    if ch_select:
        if len(ch_select) != num_of_channels:
            ch_select += ch_select[-1] * (num_of_channels - len(ch_select))

    # apply filter to each channel
    if num_of_channels == 1:
        filtered = signal.sosfiltfilt(sos, data)
    else:
        filtered = np.array(
            [
                signal.sosfiltfilt(sos, channel[0])
                for idx, channel in enumerate(data)
                if ch_select[idx] == "1"
            ]
        )

    # save results
    scipy.io.savemat(f"{output_path}.mat", {"M": filtered})

    return filtered
