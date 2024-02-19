"""Methods for creating, managing, and displaying spike trains."""

import numpy as np
import datetime
import uuid
import scipy


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


def global_spike_count_rate(spike_sequence: np.ndarray, duration: int) -> float:
    """Return the spike count rate of a spike sequence.

    :param spike_sequence: input spike sequence
    :type spike_sequence: np.ndarray
    :param duration: duration of the spike sequence
    :type duration: int

    :return: spike count rate
    :rtype: float
    """
    return num_spikes(spike_sequence) / duration


def smoothed_firing_rate(
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


class SpikeTrain:
    """Class for managing and displaying spike trains."""

    def __init__(
        self,
        spike_times: list[int] | np.ndarray,
        duration_seconds: float,
        duration_samples: int,
        sampling_period: float,
        neuron_id: int | str = None,
        recording_datetime: str | datetime.datetime = None,
    ) -> None:
        """Initialize a SpikeTrain object.

        :param spike_times: list of spike times in sample number
        :type spike_times: list[int] | np.ndarray
        :param duration_seconds: duration of the spike sequence in seconds
        :type duration_seconds: float
        :param duration_samples: duration of the spike sequence in samples
        :type duration_samples: int
        :param sampling_period: sampling period in seconds
        :type sampling_period: float
        """
        self.spike_times = spike_times
        self.spike_sequence = spike_sequence_from_spike_times(
            spike_times, sampling_period, duration_samples
        )
        self.duration_samples = duration_samples
        self.duration_seconds = duration_seconds
        self.sampling_period = sampling_period
        self.neuron_id = neuron_id
        self.recording_datetime = recording_datetime

    @property
    def num_spikes(self) -> int:
        """Return the number of spikes in the spike train."""
        return num_spikes(self.spike_sequence)

    @property
    def spike_count_rate(self) -> float:
        """Return the spike count rate of the spike train."""
        return global_spike_count_rate(self.spike_sequence, self.duration_samples)

    def smoothed_firing_rate(self, averaging_window_length: int = 3) -> np.ndarray:
        """Return the smoothed firing rate of the spike train."""
        return smoothed_firing_rate(self.spike_sequence, averaging_window_length)

    def plot(self) -> None:
        """Plot the spike train."""
        import matplotlib.pyplot as plt

        plt.eventplot(self.spike_times)
        plt.xlabel("Time (s)")
        plt.ylabel("Neuron ID = " + str(self.neuron_id))
        plt.title("Spike Train")


class SpikeTrainCollection(dict):
    """A spike train collection is a set of spike trains from multiple neurons / single neuron
    recorded multiple times. The spike trains are indexed by neuron ID and recording datetime.
    """

    def __init__(
        self,
        spike_trains: list[SpikeTrain],
        handle_missing_neuron_id: str = "raise",
    ) -> None:
        """Initialize a SpikeTrainCollection object.

        :param spike_trains: list of spike trains
        :type spike_trains: list[SpikeTrain]
        :param duration_seconds: duration of the spike sequence in seconds
        :type duration_seconds: float
        :param duration_samples: duration of the spike sequence in samples
        :type duration_samples: int
        :param sampling_period: sampling period in seconds
        :type sampling_period: float
        """
        self.spike_trains: dict[str, SpikeTrain] = {}
        for i, spike_train in enumerate(spike_trains):

            if spike_train.neuron_id is None or spike_train.recording_datetime is None:

                if handle_missing_neuron_id == "raise":
                    raise ValueError(
                        "Neuron ID and recording datetime must be provided."
                    )

                elif handle_missing_neuron_id == "assign":
                    # generate a missing neuron ID
                    neuron_id = "missing_neuron_" + str(uuid.uuid4())
                    recording_datetime = datetime.datetime.now()

            else:
                neuron_id = spike_train.neuron_id
                recording_datetime = spike_train.recording_datetime

            # index name is neuron_id + recording_datetime
            index_name = str(neuron_id) + "_" + str(recording_datetime)

            self.spike_trains[index_name] = spike_train


class MultiNeuronTrial(SpikeTrainCollection):

    def __init__(
        self,
        spike_trains: list[SpikeTrain],
        duration_seconds: float,
        duration_samples: int,
        sampling_period: float,
    ) -> None:
        """Initialize a MultiNeuronTrial object.

        :param spike_trains: list of spike trains
        :type spike_trains: list[SpikeTrain]
        :param duration_seconds: duration of the spike sequence in seconds
        :type duration_seconds: float
        :param duration_samples: duration of the spike sequence in samples
        :type duration_samples: int
        :param sampling_period: sampling period in seconds
        :type sampling_period: float
        """
        super().__init__(spike_trains)
        self.duration_seconds = duration_seconds
        self.duration_samples = duration_samples
        self.sampling_period = sampling_period

    def to_matrix(self) -> np.ndarray:
        """Return a matrix representation of the spike trains."""
        # stack all spike sequences
        spike_sequences = np.stack(
            [spike_train.spike_sequence for spike_train in self.spike_trains.values()]
        )
        return spike_sequences

    def average_firing_rate(self):
        """Return the mean firing rate across all neurons. Mean is only taken across
        neurons."""
        spike_matrix = self.to_matrix()
        return np.mean(spike_matrix, axis=0)
