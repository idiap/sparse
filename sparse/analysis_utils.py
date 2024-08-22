#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
This is where utilitary functions are defined to analyse firing rates
distributions of spiking neurons as well as cross-frequency couplings
within and across neuron populations.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
from scipy.signal import filtfilt
from scipy.signal import firwin
from scipy.signal import hilbert


def get_timings(spikes, dt):
    """
    Get timings from binary spike train with shape (num_neurons, num_steps).
    """
    neuron_indices, timings = np.where(spikes)
    timings = timings * dt
    return neuron_indices, timings


def autocorr(x, lags):
    """
    This function computes the normalized autocorrelation function.
    """
    xcorr = np.correlate(x - x.mean(), x - x.mean(), "full")
    xcorr = xcorr[xcorr.size // 2 :] / xcorr.max()
    return xcorr[: lags + 1]


def bandpass_filter(signal, freq_range, sampling_rate, num_taps=100):
    """
    Applies a finite impulse response filter to some input signal and
    acts as a bandpass filter focusing on the given frequency range.

    Arguments
    ---------
    signal : numpy array
        Input signal, p.e., a sum of spike trains from a layer.
    freq_range : list of int
        Frequency band of interest in Hz, p.e., [4, 8]Hz.
    sampling_rate : int
        Number of steps per second in the given input signal.
    num_taps : int
        Number of coefficients for the filter, i.e., filter order + 1.
    """
    fir_window = firwin(
        numtaps=num_taps,
        cutoff=freq_range,
        fs=sampling_rate,
        pass_zero=False,
        window="hamming",
    )
    return filtfilt(b=fir_window, a=1, x=signal)


def downsample_signal(data, old_fs, new_fs):
    """
    Downsamples some input signal using decimation.
    """
    factor = old_fs / new_fs
    downsampled_data = decimate(data, int(factor))
    return downsampled_data


def cross_frequency_phase_amplitude(
    phase,
    amplitude,
    num_phase_bins=18,
    phase_in_degrees=False,
):
    """
    Computes normalized average amplitude found over binned phase regions.
    Output arrays both have shape (num_phase_bins,).
    """
    if phase_in_degrees:
        phase_bins = np.linspace(-180, 180, num_phase_bins + 1)
    else:
        phase_bins = np.linspace(-np.pi, np.pi, num_phase_bins + 1)
    amp_mean = np.zeros(num_phase_bins)
    phase_mean = np.zeros(num_phase_bins)
    for k in range(num_phase_bins):
        start = phase_bins[k]
        end = phase_bins[k + 1]
        phase_mean[k] = np.mean([start, end])
        indices = (phase >= start) & (phase < end)
        if len(indices) > 0:
            amp_mean[k] = np.mean(amplitude[indices])
    amp_mean = amp_mean / np.sum(amp_mean)
    return phase_mean, amp_mean


def compute_modulation_index(amp_mean):
    """
    The input represents the average amplitude of the amplitude-providing
    frequency in each phase bin of the phase-providing frequency, with
    shape (num_phase_bins,).

    The modulation index is a measure of phase-amplitude coupling, indicating
    how much the distribution of amplitudes accross phase bins deviates from
    the uniform distribution.

    The Shannon entropy is first computed to represent the inherent amount of
    information in the input (maximal when the input is uniformly distributed
    accross bins). The Kullback–Leibler distance then measures the disparity
    between the input distribution and the uniform distribution, so that the
    modulation index is just a normalized version of it.
    """
    if len(amp_mean.shape) != 1:
        raise ValueError("The shape of amp_mean must have exactly one dimension.")
    num_phase_bins = len(amp_mean)
    shannon_entropy = -np.sum(amp_mean * np.log(amp_mean + 1e-12))
    KL_distance = np.log(num_phase_bins) - shannon_entropy
    modulation_index = KL_distance / np.log(num_phase_bins)
    return modulation_index


def compute_complex_vectors(phase, amplitude):
    """
    The 1-dim inputs represent the phase of a signal and the amplitude of a
    signal respectively. This function simply computes a complex-valued vector
    from the two inputs.
    """
    if not (len(phase.shape) == 1 and len(amplitude.shape) == 1):
        raise ValueError(
            "The shapes of phase and aplitude must have excatly one dimension."
        )
    return amplitude * np.exp(1j * phase)


def permutation_testing(
    phase,
    amplitude,
    num_surrogates=1000,
    num_phase_bins=18,
    phase_in_degrees=False,
):
    """
    The observed coupling value is compared to a distribution of surrogate
    coupling values between the original phase time series and a permuted
    amplitude time series, constructed by cutting the amplitude time series
    at a random data point and reversing the order of both parts.
    """
    num_steps = len(phase)
    mean_vector_length_surrogates = np.zeros(num_surrogates)
    modulation_index_surrogates = np.zeros(num_surrogates)

    for ns in range(num_surrogates):

        # Pick a random time step and reverse amp order on both sides
        cut_point = np.random.randint(1, num_steps)
        permuted_amp = np.concatenate(
            (
                amplitude[:cut_point][::-1],
                amplitude[cut_point:][::-1],
            )
        )

        # Compute surrogate mean vector length
        mean_vector_length_surrogates[ns] = np.abs(
            np.mean(compute_complex_vectors(phase=phase, amplitude=permuted_amp))
        )

        # Compute surrogate modulation index
        _, amp_mean_surr = cross_frequency_phase_amplitude(
            phase=phase,
            amplitude=permuted_amp,
            num_phase_bins=num_phase_bins,
            phase_in_degrees=phase_in_degrees,
        )
        modulation_index_surrogates[ns] = compute_modulation_index(
            amp_mean=amp_mean_surr
        )

    return mean_vector_length_surrogates, modulation_index_surrogates


def spikes_to_population_signal(x, time_axis_first=False):
    """
    This function aggregates input spike trains into
    a population signal with mean 0 and std 1.
    """
    neuron_axis = 1 if time_axis_first else 0
    x = x.mean(axis=neuron_axis)
    x = normalize_signal(x)
    return x


def compute_pac(
    lowfreq_signal,
    highfreq_signal,
    lowfreq_signal_type="spikes",
    highfreq_signal_type="spikes",
    lowfreq_range=[3, 8],
    highfreq_range=[30, 80],
    sampling_rate_spikes=1000,
    sampling_rate_waveform=16000,
    num_phase_bins=18,
    num_surrogates=10000,
    time_axis_first=False,
    phase_in_degrees=False,
):
    """
    This function takes two spike train arrays as inputs that will be aggregated
    into two population signals. The first will be filtered in the given
    lowfreq_range and the second in the highfreq_range. Mean vector length and
    modulation index will be computed to quantify the modulation of the
    amplitude of the second by the phase of the first. The same metrics will be
    computed for num_surrogates permutations to assess the significant of the
    observed values.
    The function returns the observed and surrogate values along with some
    useful quantities for further analysis and plots.
    """

    # Filter low-frequency signal
    if lowfreq_signal_type == "spikes":
        if len(lowfreq_signal.shape) != 2:
            raise ValueError(
                "The shape of spikes lowfreq_signal must have exactly 2 dimensions."
            )
        lowfreq_signal = spikes_to_population_signal(
            x=lowfreq_signal, time_axis_first=time_axis_first
        )
        lowfreq_filtered = bandpass_filter(
            signal=lowfreq_signal,
            freq_range=lowfreq_range,
            sampling_rate=sampling_rate_spikes,
        )
    elif lowfreq_signal_type == "waveform":
        if len(lowfreq_signal.shape) != 1:
            raise ValueError(
                "The shape of waveform lowfreq_signal must have exactly one dimension."
            )
        lowfreq_filtered = bandpass_filter(
            signal=lowfreq_signal,
            freq_range=lowfreq_range,
            sampling_rate=sampling_rate_waveform,
        )
        lowfreq_filtered = downsample_signal(
            data=lowfreq_filtered,
            old_fs=sampling_rate_waveform,
            new_fs=sampling_rate_spikes,
        )
    else:
        raise ValueError("lowfreq_signal_type must be spikes or waveform.")

    # Compute phase
    lowfreq_phase = np.angle(hilbert(lowfreq_filtered))
    if phase_in_degrees:
        lowfreq_phase = np.degrees(lowfreq_phase)

    # Filter high-frequency signal
    if highfreq_signal_type == "spikes":
        if len(highfreq_signal.shape) != 2:
            raise ValueError(
                "The shape of spikes highfreq_signal must have exactly 2 dimensions."
            )
        highfreq_signal = spikes_to_population_signal(
            x=highfreq_signal, time_axis_first=time_axis_first
        )
        highfreq_filtered = bandpass_filter(
            signal=highfreq_signal,
            freq_range=highfreq_range,
            sampling_rate=sampling_rate_spikes,
        )

    elif highfreq_signal_type == "waveform":
        if len(highfreq_signal.shape) != 1:
            raise ValueError(
                "The shape of waveform highfreq_signal must have exactly one dimension."
            )
        highfreq_filtered = bandpass_filter(
            signal=highfreq_signal,
            freq_range=highfreq_range,
            sampling_rate=sampling_rate_waveform,
        )
        highfreq_filtered = downsample_signal(
            data=highfreq_filtered,
            old_fs=sampling_rate_waveform,
            new_fs=sampling_rate_spikes,
        )
    else:
        raise ValueError("amplitude_signal_type must be spikes or waveform.")

    # Compute amplitude
    highfreq_amp = np.abs(hilbert(highfreq_filtered))

    # Ensure same number of time steps
    if len(lowfreq_phase) > len(highfreq_amp):
        lowfreq_phase = lowfreq_phase[: len(highfreq_amp)]
    elif len(lowfreq_phase) < len(highfreq_amp):
        highfreq_amp = highfreq_amp[: len(lowfreq_phase)]

    # Compute observed mean vector length
    complex_vectors_observed = compute_complex_vectors(
        phase=lowfreq_phase,
        amplitude=highfreq_amp,
    )
    mean_vector_length_observed = np.abs(np.mean(complex_vectors_observed))

    # Compute observed cross frequency phase-amplitude relation
    phase_mean_obs, amp_mean_obs = cross_frequency_phase_amplitude(
        phase=lowfreq_phase,
        amplitude=highfreq_amp,
        num_phase_bins=num_phase_bins,
        phase_in_degrees=phase_in_degrees,
    )

    # Compute observed modulation index
    modulation_index_observed = compute_modulation_index(amp_mean_obs)

    # Test significance by generating surrogates with resampled amplitudes
    mean_vector_length_surrogates, modulation_index_surrogates = permutation_testing(
        phase=lowfreq_phase,
        amplitude=highfreq_amp,
        num_surrogates=num_surrogates,
        num_phase_bins=num_phase_bins,
        phase_in_degrees=phase_in_degrees,
    )

    return (
        mean_vector_length_observed,
        modulation_index_observed,
        mean_vector_length_surrogates,
        modulation_index_surrogates,
        complex_vectors_observed,
        phase_mean_obs,
        amp_mean_obs,
        lowfreq_signal,
        highfreq_signal,
        lowfreq_phase,
    )


def compute_pvalue(surrogate_values, observed_value):
    num_surrogates = len(surrogate_values)
    count_greater_or_equal = np.sum(surrogate_values >= observed_value)
    p_value = count_greater_or_equal / num_surrogates
    return p_value


def print_significance(p_value, threshold=0.05):
    is_significant = "is" if p_value < threshold else "is NOT"
    print(f"p-value={p_value}, which {is_significant} significant.")


def write_significance_mvl_mi(
    pvalues_mvl,
    pvalues_mi,
    layer_combinations,
    lowfreq_names,
    highfreq_names,
    save_folder,
):
    """
    This function writes the significant measured couplings to a text file.
    """
    if not (len(pvalues_mvl.shape) == 4 and len(pvalues_mi.shape) == 4):
        raise ValueError(
            "The shape of pvalues_mvl and p_values_mi must have exactly 4 dimensions."
        )
    num_utterances = pvalues_mvl.shape[0]

    # Write to file
    with open(f"{save_folder}/pac_analysis.txt", "w") as file:
        file.write(
            "\n--------- PAC ANALYSIS ---------\n"
            "\n\nIn the following analysis, layer index 0 represents the input "
            "waveform, index 1, spikes produced by the auditory nerve fibers,"
            "index 2 to the first layer of the multilayered SNN etc.\n\n"
        )

        # Significant MVL or MI
        num_pvalues_mvl = (pvalues_mvl < 0.05).sum()
        num_pvalues_mi = (pvalues_mi < 0.05).sum()
        file.write(
            f"There were {num_pvalues_mvl} pvalues < 0.05 for MVL.\n"
            f"There were {num_pvalues_mi} pvalues < 0.05 for MI.\n"
        )

        # Significant MVL and MI on a single utterance
        ids = np.where(((pvalues_mvl < 0.05) * (pvalues_mi < 0.05)))
        num_significant = len(ids[0])
        file.write(
            f"\nOn a single utterance, there were {num_significant} pvalues < "
            f"0.05 for MVL and MI:\n"
        )
        for i in range(num_significant):
            w, x = ids[0][i], ids[1][i]
            y, z = ids[2][i], ids[3][i]
            file.write(
                f"utt-id={w}, {layer_combinations[x]}: "
                f"{lowfreq_names[y]} {highfreq_names[z]}\n"
            )

        # Significant MVL and MI on more than one utterance
        for n_utt in range(1, num_utterances + 1):
            ids = np.where(((pvalues_mvl < 0.05) * (pvalues_mi < 0.05)).sum(0) > n_utt)
            num_significant = len(ids[0])

            if num_significant == 0:
                break

            file.write(
                f"\nOn more than {n_utt} utterances there were {num_significant}"
                f" pvalues < 0.05 for MVL and MI:\n"
            )
            for i in range(num_significant):
                x, y, z = ids[0][i], ids[1][i], ids[2][i]
                file.write(
                    f"{layer_combinations[x]}: "
                    f"{lowfreq_names[y]} {highfreq_names[z]}\n"
                )

        # List number of utterances with each form of coupling
        file.write("\nList of all couplings with number of utterances having it:\n")
        num_significants = ((pvalues_mvl < 0.05) * (pvalues_mi < 0.05)).sum(0)
        for x in range(pvalues_mvl.shape[1]):
            for y in range(pvalues_mvl.shape[2]):
                for z in range(pvalues_mvl.shape[3]):
                    num_significant = num_significants[x, y, z]
                    file.write(
                        f"{layer_combinations[x]}: {lowfreq_names[y]}"
                        f" {highfreq_names[z]} -> {num_significant} \n"
                    )

    print(f"\nAnalysis written to {save_folder}/pac_analysis.txt\n")


def gauss_dist(x, coeff, mu, sig):
    return coeff * np.exp(-0.5 * (x - mu) ** 2 / sig**2)


def scale_signal(signal, range=[-1, 1]):
    if range == [-1, 1]:
        return 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1
    elif range == [0, 1]:
        return (signal - signal.min()) / (signal.max() - signal.min())
    else:
        raise NotImplementedError


def normalize_signal(signal):
    return (signal - signal.mean()) / signal.std()


def show_or_save(out_name, save_folder, save_plot=False):
    if save_plot:
        plt.savefig(f"{save_folder}/{out_name}", dpi=300)
        plt.close()
        print(f"\nPlot created at {save_folder}/{out_name}.\n")
    else:
        plt.show()
        plt.close()
