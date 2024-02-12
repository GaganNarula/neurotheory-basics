"""Methods for creating, managing, and displaying spike trains."""

import numpy as np


def spike_sequence_from_spike_times(
    spike_times: list[int] | np.ndarray,
    sampling_period: float | None = None,
    duration_samples: int | None = None,
    duration_seconds: float | None = None,
) -> np.ndarray:
    """Create a spike sequence from a list of spike times.

    :param spike_times: list of spike times in sample number
    :type spike_times: list[int] | np.ndarray
    :param sampling_period: sampling period in seconds
    :type sampling_period: float, optional
    :param duration_samples: duration of the spike sequence in samples
    :type duration_samples: int, optional
    :param duration_seconds: duration of the spike sequence in seconds
    :type duration_seconds: float, optional

    :return: spike sequence
    :rtype: np.ndarray
    """
    if duration_samples is not None:
        duration = duration_samples
    elif duration_seconds is not None:
        duration = int(duration_seconds / sampling_period)
    else:
        raise ValueError(
            "Either duration_samples or duration_seconds must be provided."
        )

    spike_sequence = np.zeros(duration)
    spike_sequence[spike_times] = 1
    return spike_sequence


def num_spikes(spike_sequence: np.ndarray) -> int:
    """Return the number of spikes in a spike sequence.

    :param spike_sequence: input spike sequence
    :type spike_sequence: np.ndarray

    :return: number of spikes
    :rtype: int
    """
    return np.sum(spike_sequence)


def spike_count_rate(spike_sequence: np.ndarray, duration: int) -> float:
    """Return the spike count rate of a spike sequence.

    :param spike_sequence: input spike sequence
    :type spike_sequence: np.ndarray
    :param duration: duration of the spike sequence
    :type duration: int

    :return: spike count rate
    :rtype: float
    """
    return num_spikes(spike_sequence) / duration
