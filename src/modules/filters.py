"""Contains methods for filtering spike trains"""

import numpy as np
import scipy


def boxcar_smoother(
    spike_sequence: np.ndarray, averaging_window_length: int = 3, mode: str = "same"
) -> np.ndarray:
    """Smoothed firing rate of a single spike sequence. The smoothed firing rate is
    obtained by convolving the spike sequence with a rectangular window.

    :param spike_sequence: input spike sequence
    :type spike_sequence: np.ndarray
    :param averaging_window_length: length of the averaging window
    :type averaging_window_length: int, optional
    :param mode: mode of the convolution
    :type mode: str, optional

    :return: smoothed firing rate
    :rtype: np.ndarray
    """
    return scipy.signal.convolve(
        spike_sequence,
        np.ones(averaging_window_length) / averaging_window_length,
        mode=mode,
    )


def gaussian_smoother(
    spike_sequence: np.ndarray, sigma: float = 1, mode: str = "same"
) -> np.ndarray:
    """Smoothed firing rate of a single spike sequence. The smoothed firing rate is
    obtained by convolving the spike sequence with a Gaussian window.

    :param spike_sequence: input spike sequence
    :type spike_sequence: np.ndarray
    :param sigma: standard deviation of the Gaussian window
    :type sigma: float, optional
    :param mode: mode of the convolution
    :type mode: str, optional

    :return: smoothed firing rate
    :rtype: np.ndarray
    """
    return scipy.ndimage.gaussian_filter1d(spike_sequence, sigma, mode=mode)


def alpha_function_smoother(spike_sequence: np.ndarray, alpha: float = 1) -> np.ndarray:
    """Smoothed firing rate of a single spike sequence. The smoothed firing rate is
    obtained by convolving the spike sequence with an alpha function.

    :param spike_sequence: input spike sequence
    :type spike_sequence: np.ndarray
    :param alpha: time constant of the alpha function
    :type alpha: float, optional

    :return: smoothed firing rate
    :rtype: np.ndarray
    """
    t = np.arange(len(spike_sequence))  # time
    kernel = (alpha**2) * t * np.exp(-alpha * t)
    kernel = np.max(kernel, 0)  # make sure the kernel is causal

    return scipy.signal.convolve(spike_sequence, kernel, mode="same")


class SpikeTrainFilter:
    """Filter spike trains with different convolution kernels"""

    def __init__(self, kernel: str = "boxcar", **kwargs):
        """Initialize the spike train filter

        :param kernel: convolution kernel, defaults to "boxcar"
        :type kernel: str, optional
        """
        self.kernel = kernel
        self.kwargs = kwargs

    def filter(self, spike_sequence: np.ndarray) -> np.ndarray:
        """Filter the spike sequence

        :param spike_sequence: input spike sequence
        :type spike_sequence: np.ndarray

        :return: filtered spike sequence
        :rtype: np.ndarray
        """
        if self.kernel == "boxcar":
            return boxcar_smoother(spike_sequence, **self.kwargs)
        elif self.kernel == "gaussian":
            return gaussian_smoother(spike_sequence, **self.kwargs)
        else:
            raise ValueError(f"Kernel {self.kernel} not supported")
