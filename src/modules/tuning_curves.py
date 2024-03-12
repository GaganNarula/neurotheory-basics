"""This module contains different types of tuning curves for neuron responses."""

import numpy as np


def gaussian_tuning_curve(
    input_value: np.ndarray | float,
    preferred_value: np.ndarray | float,
    max_firing_rate: np.ndarray | float,
    tuning_width: np.ndarray | float,
) -> np.ndarray | float:
    """Calculate the firing rate of a neuron with a Gaussian tuning curve.

    :param input_value: The input value to the neuron.
    :type input_value: np.ndarray | float
    :param preferred_value: The preferred value of the neuron.
    :type preferred_value: np.ndarray | float
    :param max_firing_rate: The maximum firing rate of the neuron.
    :type max_firing_rate: np.ndarray | float
    :param tuning_width: The tuning width of the neuron.
    :type tuning_width: np.ndarray | float
    :return: The firing rate of the neuron.
    :rtype: np.ndarray | float
    """
    return max_firing_rate * np.exp(
        -0.5 * ((input_value - preferred_value) / tuning_width) ** 2
    )


def sigmoid_tuning_curve(
    input_value: np.ndarray | float,
    value_at_half_max: np.ndarray | float,
    slope: np.ndarray | float,
    max_firing_rate: np.ndarray | float,
) -> np.ndarray | float:
    """Calculate the firing rate of a neuron with a sigmoid tuning curve.

    :param input_value: The input value to the neuron.
    :type input_value: np.ndarray | float
    :param value_at_half_max: The value at half maximum firing rate. Similar to the preferred value.
    :type value_at_half_max: np.ndarray | float
    :param slope: The slope of the sigmoid.
    :type slope: np.ndarray | float
    :param max_firing_rate: The maximum firing rate of the neuron.
    :type max_firing_rate: np.ndarray | float
    :return: The firing rate of the neuron.
    :rtype: np.ndarray | float
    """
    return max_firing_rate / (1 + np.exp(-slope * (input_value - value_at_half_max)))
