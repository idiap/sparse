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
within or across neuron populations.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.stats import norm
from tqdm import tqdm

from sparse.analysis_utils import bandpass_filter
from sparse.analysis_utils import compute_pac
from sparse.analysis_utils import compute_pvalue
from sparse.analysis_utils import cross_frequency_phase_amplitude
from sparse.analysis_utils import downsample_signal
from sparse.analysis_utils import gauss_dist
from sparse.analysis_utils import normalize_signal
from sparse.analysis_utils import show_or_save
from sparse.analysis_utils import write_significance_mvl_mi


def analyse_spikes(
    model,
    all_spikes,
    feats,
    wavs,
    wav_lens,
    hop_length_ms,
    phase_layer=3,
    amplitude_layer=3,
    phase_freq_range=[4, 8],
    amplitude_freq_range=[30, 80],
    phase_band_name="Theta",
    amplitude_band_name="Low-Gamma",
    num_utterances_pac_analysis=64,
    utterance_id=-1,
    plot_parameters=False,
    plot_weights=False,
    plot_spikes=True,
    plot_rates=True,
    plot_phase_amplitude_coupling=True,
    plot_filtered_population_signal=True,
    run_pac_analysis=False,
    save_plots=True,
    save_folder="results/Analysis_2ms_16chan_3x512_ff1_rnn5_sfa5/plots/",
):
    """
    Performs analysis on spike trains all_spikes, organised as a list of
    num_layers tensors with shape (batch_size, num_steps, num_neurons)
    where num_neurons can vary across layers. Plots are saved at given
    location save_folder.
    """

    # Create folder where plots are saved
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Define useful variables
    num_layers = len(all_spikes)
    num_steps = all_spikes[0].shape[1]
    dt = hop_length_ms * 1e-3
    sampling_rate = 1 / dt
    all_spikes = [spikes.detach().cpu().numpy() for spikes in all_spikes]
    feats = feats.detach().cpu().numpy()
    wavs = wavs.detach().cpu().numpy()
    wav_lens = wav_lens.detach().cpu().numpy()
    utt_steps = int(num_steps * wav_lens[utterance_id])
    utt_duration = utt_steps * dt

    # Last batch element is gaussian noise instead of speech features
    if utterance_id in [-1, len(wav_lens)]:
        input_type = "gaussian_noise"
    else:
        input_type = f"utt{utterance_id}"

    # Plot neuron parameter distributions
    if plot_parameters:
        _, axes = plt.subplots(nrows=num_layers, ncols=4)
        for i in range(num_layers):
            if i == 0:
                nonzero = torch.where(model.snn_auditory.sfa_mask)[0]
                tauu = model.snn_auditory.tauu
                tauw = model.snn_auditory.tauw[nonzero]
                a = model.snn_auditory.a[nonzero]
                b = model.snn_auditory.b[nonzero]
            else:
                nonzero = torch.where(
                    model.snn_cortex.snn[i - 1].spiking_neurons.sfa_mask
                )[0]
                tauu = model.snn_cortex.snn[i - 1].spiking_neurons.tauu
                tauw = model.snn_cortex.snn[i - 1].spiking_neurons.tauw[nonzero]
                a = model.snn_cortex.snn[i - 1].spiking_neurons.a[nonzero]
                b = model.snn_cortex.snn[i - 1].spiking_neurons.b[nonzero]

            axes[i, 0].hist(tauu.detach().cpu().numpy(), bins=10, edgecolor="black")
            axes[i, 1].hist(tauw.detach().cpu().numpy(), bins=10, edgecolor="black")
            axes[i, 2].hist(a.detach().cpu().numpy(), bins=10, edgecolor="black")
            axes[i, 3].hist(b.detach().cpu().numpy(), bins=10, edgecolor="black")

            axes[i, 0].set_title(f"tauu, layer={i}", fontsize=6)
            axes[i, 1].set_title(f"tauw, layer={i}", fontsize=6)
            axes[i, 2].set_title(f"a, layer={i}", fontsize=6)
            axes[i, 3].set_title(f"b, layer={i}", fontsize=6)

        for ax in axes.ravel():
            ax.tick_params(axis="both", which="both", labelsize=4)

        plt.tight_layout()
        plt.savefig(f"{save_folder}/parameters.png", dpi=300)
        print(f"\nPlot created at {save_folder}/parameters.png\n")

    # Plot weight ditribution
    if plot_weights:
        _, axes = plt.subplots(nrows=num_layers - 1, ncols=2)
        for i in range(num_layers - 1):

            # Get matrix parameters ignoring masked weights
            nonzero = torch.where(
                model.snn_cortex.snn[i].spiking_neurons.rnn_mask.flatten()
            )[0]
            W = model.snn_cortex.snn[i].W.weight.flatten()
            V = model.snn_cortex.snn[i].spiking_neurons.V.weight.flatten()[nonzero]

            # Proportion of positive/negative weights
            p_excit_W_weights = (W > 0).float().mean().detach().cpu().numpy()
            p_excit_V_weights = (V > 0).float().mean().detach().cpu().numpy()

            # Plot histograms
            axes[i, 0].hist(W.detach().cpu().numpy(), bins=1000)
            axes[i, 1].hist(V.detach().cpu().numpy(), bins=1000)

            # Add vertical line at zero
            axes[i, 0].axvline(x=0, color="r", linestyle="--")
            axes[i, 1].axvline(x=0, color="r", linestyle="--")

            # Add text annotations for proportions
            axes[i, 0].text(
                0.05,
                0.95,
                f"{1-p_excit_W_weights:.2f}",
                transform=axes[i, 0].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                color="blue",
            )
            axes[i, 0].text(
                0.95,
                0.95,
                f"{p_excit_W_weights:.2f}",
                transform=axes[i, 0].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                color="darkorange",
            )
            axes[i, 1].text(
                0.05,
                0.95,
                f"{1-p_excit_V_weights:.2f}",
                transform=axes[i, 1].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                color="blue",
            )
            axes[i, 1].text(
                0.95,
                0.95,
                f"{p_excit_V_weights:.2f}",
                transform=axes[i, 1].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                color="darkorange",
            )

            # Set y-labels
            axes[i, 0].set_ylabel(f"W, layer {i}", fontsize=12)
            axes[i, 1].set_ylabel(f"V, layer {i}", fontsize=12)

            # Hide y-ticks
            axes[i, 0].set_yticks([])
            axes[i, 1].set_yticks([])

        for ax in axes.ravel():
            ax.tick_params(axis="both", which="both", labelsize=10)

        plt.tight_layout()
        plt.savefig(f"{save_folder}/weights.png", dpi=300)
        print(f"\nPlot created at {save_folder}/weights.png\n")

    # Plot freq-ordered spike trains
    if plot_spikes:
        _, axes = plt.subplots(nrows=num_layers + 1, ncols=1)
        ticks = np.linspace(0, num_steps - 1, 5)
        ticklabels = np.round(ticks * dt, 1)

        for i, ax in enumerate(axes.ravel()):
            if i == 0:
                ax.imshow(
                    feats[utterance_id, :utt_steps].T,
                    aspect="auto",
                    origin="lower",
                    interpolation="bicubic",
                )
                ax.set_ylabel("Input", fontsize=8)
            else:
                freqs = all_spikes[i - 1][utterance_id, :utt_steps, :].mean(axis=0)
                ordered_index = np.argsort(freqs)
                # ordered_index = np.arange(len(freqs)) # not sorted
                ax.imshow(
                    all_spikes[i - 1][utterance_id, :utt_steps, ordered_index],
                    aspect="auto",
                    interpolation="none",
                )
                if i == 1:
                    ax.set_ylabel("Nerve", fontsize=8)
                else:
                    ax.set_ylabel(f"Layer {i-1}", fontsize=8)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_xlim(0, utt_steps)
        ax.set_xlabel("Time (s)", fontsize=8)
        plt.savefig(f"{save_folder}/spikes_ordered_{input_type}.png", dpi=300)
        print(f"\nPlot created at {save_folder}/spikes_ordered_{input_type}.png\n")

    # Plot firing rate distribution
    if plot_rates:
        firing_rates(
            all_spikes=all_spikes,
            duration=utt_duration,
            num_steps=utt_steps,
            input_type=input_type,
            bin_width=int(10 / hop_length_ms),
            utterance_id=utterance_id,
            sampling_rate=sampling_rate,
            use_single_utterance=True,
            save_plot=save_plots,
            save_folder=save_folder,
        )

    # Use a single utterance for rest of analysis
    _spikes = []
    for i in range(num_layers):
        _spikes.append(all_spikes[i][utterance_id, :utt_steps].transpose())

    # Plot population signals filtered in different bands
    if plot_filtered_population_signal:
        filter_population_signal(
            spike_trains_A=_spikes[phase_layer],
            spike_trains_B=_spikes[amplitude_layer],
            layer_A=phase_layer,
            layer_B=amplitude_layer,
            input_type=input_type,
            sampling_rate=sampling_rate,
            time_axis_first=False,
            save_plots=save_plots,
            save_folder=save_folder,
        )

    # Plot phase-amplitude coupling
    if plot_phase_amplitude_coupling:

        # Create subfolder based on arguments
        pac_folder = save_folder + f"/pac_layer{phase_layer}_{amplitude_layer}_"
        pac_folder += f"freq{phase_freq_range[0]}_{phase_freq_range[1]}_"
        pac_folder += f"{amplitude_freq_range[0]}_{amplitude_freq_range[1]}_"
        pac_folder += input_type
        if not os.path.exists(pac_folder):
            os.makedirs(pac_folder)

        pac_plots(
            lowfreq_signal=_spikes[phase_layer],
            highfreq_signal=_spikes[amplitude_layer],
            lowfreq_signal_type="spikes",
            highfreq_signal_type="spikes",
            lowfreq_range=phase_freq_range,
            highfreq_range=amplitude_freq_range,
            lowfreq_name=phase_band_name,
            highfreq_name=amplitude_band_name,
            sampling_rate_spikes=sampling_rate,
            sampling_rate_waveform=16000,
            time_axis_first=False,
            phase_in_degrees=False,
            save_plots=save_plots,
            save_folder=pac_folder,
        )

    # Run PAC analysis across all layers and bands
    if run_pac_analysis:
        (pvalues_mvl, pvalues_mi, layer_combinations, lowfreq_names, highfreq_names) = (
            pac_analysis(
                all_spikes=all_spikes,
                waveforms=wavs,
                wav_lens=wav_lens,
                num_utterances=num_utterances_pac_analysis,
                sampling_rate_spikes=sampling_rate,
                sampling_rate_waveform=16000,
                num_surrogates=1000,
            )
        )
        write_significance_mvl_mi(
            pvalues_mvl=pvalues_mvl,
            pvalues_mi=pvalues_mi,
            layer_combinations=layer_combinations,
            lowfreq_names=lowfreq_names,
            highfreq_names=highfreq_names,
            save_folder=save_folder,
        )


def firing_rates(
    all_spikes,
    duration,
    num_steps,
    input_type,
    bin_width=4,
    utterance_id=0,
    sampling_rate=1000,
    use_single_utterance=True,
    save_plot=True,
    save_folder=".",
):
    """
    Plot histogram of individual neuron firing rates for each layer.
    Distinguish between different brain rhythms.
    """

    # Whether to use single utterance or whole batch
    if use_single_utterance:
        all_rates = [
            np.sum(x[utterance_id, :num_steps], axis=0) / duration for x in all_spikes
        ]
    else:
        all_rates = [np.mean(np.sum(x, axis=1) / duration, axis=0) for x in all_spikes]
    max_rate = min(np.concatenate(all_rates, axis=-1).max(), sampling_rate // 2)
    rate_bins = [0, 0.5] + list(range(1, int(max_rate), bin_width))

    # Define ranges for different brain rhythms
    rhythm_ranges = [0.5, 4, 8, 13, 30, 70, 150, int(max_rate)]
    rhythm_names = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "low-gamma",
        "high-gamma",
        "higher",
    ]
    rhythm_colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
    layer_names = ["Nerve"]
    for i in range(len(all_spikes) - 1):
        layer_names.append(f"Layer {i+1}")

    # Remove rates > 150 Hz if time step too large
    if max_rate < 150:
        rhythm_ranges = rhythm_ranges[:-1]
        rhythm_names = rhythm_names[:-1]
        rhythm_colors = rhythm_colors[:-1]

    # Subplots
    _, axes = plt.subplots(nrows=len(all_spikes), ncols=1)
    plt.setp(axes, xlim=(0, max_rate))

    for i, ax in enumerate(axes.ravel()):
        hist_vals, _, _ = ax.hist(
            all_rates[i],
            bins=rate_bins,
            edgecolor="black",
        )
        ymax = hist_vals[1:].max() + 5
        avg_rate = all_rates[i].mean()
        ax.set_ylabel(layer_names[i])
        ax.vlines(x=avg_rate, ymin=0, ymax=ymax, color="r", label="average rate")
        ax.set_ylim(0, ymax)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        for j in range(len(rhythm_names)):
            current_label = (
                f"{rhythm_names[j]} {rhythm_ranges[j]}-{rhythm_ranges[j+1]} Hz"
            )
            ax.fill_between(
                x=np.linspace(rhythm_ranges[j], rhythm_ranges[j + 1], 10),
                y1=ymax,
                facecolor=rhythm_colors[j],
                label=current_label,
                alpha=0.3,
            )

    axes[-1].legend(
        prop={"size": 10}, loc="lower right", bbox_to_anchor=(1, 0), borderaxespad=0
    )
    axes[-1].set_xlabel("Single neuron firing rate (Hz)", fontsize=10)
    # axes[2].set_ylabel("Neuron count (a.u.)", fontsize=10)
    axes[-1].xaxis.set_ticks(list(range(0, int(max_rate), 50)))
    axes[-1].xaxis.set_ticklabels(list(range(0, int(max_rate), 50)))
    plt.tight_layout(w_pad=0.1)
    plt.subplots_adjust(hspace=0.1)
    show_or_save(
        f"rates_{input_type}.png", save_folder=save_folder, save_plot=save_plot
    )


def pac_plots(
    lowfreq_signal,
    highfreq_signal,
    lowfreq_signal_type="spikes",
    highfreq_signal_type="spikes",
    lowfreq_range=[3, 8],
    highfreq_range=[30, 80],
    lowfreq_name="Theta",
    highfreq_name="Low-Gamma",
    sampling_rate_spikes=1000,
    sampling_rate_waveform=16000,
    num_phase_bins=18,
    num_surrogates=10000,
    time_axis_first=True,
    phase_in_degrees=True,
    save_plots=False,
    save_folder=".",
):
    """
    Compute cross-frequency Phase-Amplitude Coupling (PAC) for two arrays
    of binary spike trains with shape (num_steps, num_neurons).

    Arguments
    ---------
    lowfreq_signal : numpy array
        Either binary spike train with shape (num_steps, num_neurons) for low
        frequency oscillations, or waveform with shape (num_steps,).
    highfreq_signal : numpy array
        Either binary spike train with shape (num_steps, num_neurons) for high
        frequency oscillations, or waveform with shape (num_steps,).
    lowfreq_signal_type, highfreq_signal_type: str, str
        Type of signal "spikes" or "waveform".
    lowfreq_range : list
        Frequency range for the lowfreq oscillations (e.g., [3, 8] Hz).
    highfreq_range : float list
        Frequency range for the highfreq oscillations (e.g., [30, 80] Hz).
    sampling_rate_spikes : int
        Sampling rate of the spike trains in Hz.
    num_phase_bins : int
        Number of bins to split the range of phase values (-pi, pi).
    time_axis_first : bool
        Whether the inputs have shape (steps, neurons) or (neurons, steps).
    phase_in_degrees : bool
        Whether to use degrees or radians for the phase.
    save_plots : bool
        Whether to display the produced plots or not.
    """
    # Compute PAC
    (
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
    ) = compute_pac(
        lowfreq_signal=lowfreq_signal,
        highfreq_signal=highfreq_signal,
        lowfreq_signal_type=lowfreq_signal_type,
        highfreq_signal_type=highfreq_signal_type,
        lowfreq_range=lowfreq_range,
        highfreq_range=highfreq_range,
        sampling_rate_spikes=sampling_rate_spikes,
        sampling_rate_waveform=sampling_rate_waveform,
        num_phase_bins=num_phase_bins,
        num_surrogates=num_surrogates,
        time_axis_first=time_axis_first,
        phase_in_degrees=phase_in_degrees,
    )

    # Prepare phase-freq 3D plot
    highfreq_freqs = np.arange(highfreq_range[0], highfreq_range[1] + 1)
    amp_mean_over_freqs = np.zeros((len(highfreq_freqs), len(phase_mean_obs)))
    for i, freq in enumerate(highfreq_freqs):
        if highfreq_signal_type == "spikes":
            _highfreq_filtered = bandpass_filter(
                signal=highfreq_signal,
                freq_range=[freq, freq + 1],
                sampling_rate=sampling_rate_spikes,
            )
        elif highfreq_signal_type == "waveform":
            _highfreq_filtered = bandpass_filter(
                signal=highfreq_signal,
                freq_range=[freq, freq + 1],
                sampling_rate=sampling_rate_waveform,
            )
            _highfreq_filtered = downsample_signal(
                data=_highfreq_filtered,
                old_fs=sampling_rate_waveform,
                new_fs=sampling_rate_spikes,
            )
        _highfreq_amp = np.abs(hilbert(_highfreq_filtered))
        _phase_mean, _amp_mean = cross_frequency_phase_amplitude(
            phase=lowfreq_phase,
            amplitude=_highfreq_amp,
            num_phase_bins=num_phase_bins,
            phase_in_degrees=phase_in_degrees,
        )
        amp_mean_over_freqs[i, :] = _amp_mean

    # Subplot 1.1: Amplitude distribution over phase bins
    angle_type = "degrees" if phase_in_degrees else "radians"
    angle_range = 360 if phase_in_degrees else 2 * np.pi
    phase_bin_width = angle_range / num_phase_bins

    plt.figure()
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar(phase_mean_obs, amp_mean_obs, width=phase_bin_width, edgecolor="black")
    ax1.set_ylabel(f"{highfreq_name} amplitude")
    ax1.set_xlabel(f"{lowfreq_name} phase ({angle_type})")

    if not phase_in_degrees:
        xticks_values = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        xticks_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
        ax1.set_xticks(xticks_values)
        ax1.set_xticklabels(xticks_labels)

    # Subplot 1.2: Modulation index surrogates versus observed
    ax2 = plt.subplot(gs[0, 1])
    counts, bins, _ = ax2.hist(
        modulation_index_surrogates, label="surrogates", edgecolor="black"
    )
    ax2.vlines(
        modulation_index_observed, 0, max(counts), colors="red", label="observed", lw=3
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    initial_guess = [1.0, np.mean(bin_centers), np.std(bin_centers)]
    params, _ = curve_fit(gauss_dist, bin_centers, counts, p0=initial_guess)
    xmax = max(3 * modulation_index_observed, bins.max())
    amplitude_fit, mean_fit, std_dev_fit = params
    x_fit = np.linspace(min(bins), xmax, 100)
    y_fit = gauss_dist(x_fit, amplitude_fit, mean_fit, std_dev_fit)
    z_score = (modulation_index_observed - mean_fit) / abs(std_dev_fit)
    p_value = 1 - norm.cdf(z_score)
    ax2.plot(x_fit, y_fit, "--", label="Gaussian fit")
    ax2.text(1.1 * modulation_index_observed, np.mean(counts), f"$p$={p_value:0.2e}")
    ax2.set_xlim(bins.min(), xmax)
    ax2.legend(loc="upper right")
    ax2.set_xlabel("Modulation index")
    ax2.set_ylabel("Number of surrogates")

    # Subplot 1.3: Amplitude and phase distribution in polar coordinates
    ax3 = plt.subplot(gs[1, 0], polar=True)
    rmax = np.max(np.abs(complex_vectors_observed))
    ax3.set_rmax(rmax)
    ax3.scatter(
        np.angle(complex_vectors_observed),
        np.abs(complex_vectors_observed),
        marker="o",
        label="complex values",
    )
    ax3.set_xlabel(f"{lowfreq_name} phase (radians)")
    ax3.set_ylabel(f"{highfreq_name} amplitude", labelpad=20)
    ax3.tick_params(axis="y", labelsize=8)
    ax3.yaxis.set_major_locator(MaxNLocator(4))
    if not phase_in_degrees:
        ax3.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax3.set_xticklabels(
            ["0", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"], position=(0.5, 0.07)
        )

    # Subplot 1.4: Mean vector length surrogates versus observed
    ax4 = plt.subplot(gs[1, 1])
    counts, bins, _ = ax4.hist(
        mean_vector_length_surrogates, label="surrogates", edgecolor="black"
    )
    ax4.vlines(
        mean_vector_length_observed,
        0,
        max(counts),
        colors="red",
        label="observed",
        lw=3,
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    initial_guess = [1.0, np.mean(bin_centers), np.std(bin_centers)]
    params, _ = curve_fit(gauss_dist, bin_centers, counts, p0=initial_guess)
    xmax = max(3 * mean_vector_length_observed, bins.max())
    amplitude_fit, mean_fit, std_dev_fit = params
    x_fit = np.linspace(min(bins), xmax, 100)
    y_fit = gauss_dist(x_fit, amplitude_fit, mean_fit, std_dev_fit)
    z_score = (mean_vector_length_observed - mean_fit) / abs(std_dev_fit)
    p_value = 1 - norm.cdf(z_score)
    ax4.plot(x_fit, y_fit, "--", label="Gaussian fit")
    ax4.text(1.1 * mean_vector_length_observed, np.mean(counts), f"$p$={p_value:0.2e}")
    ax4.set_xlim(bins.min(), xmax)
    ax4.set_xlabel("Mean vector length")
    ax4.set_ylabel("Number of surrogates")

    plt.tight_layout()
    show_or_save("pac.png", save_folder=save_folder, save_plot=save_plots)

    # Plot 2: lowfreq phase - highfreq freq
    plt.figure()
    plt.title(f"{lowfreq_name} phase - {highfreq_name} freq")
    plt.imshow(
        amp_mean_over_freqs,
        extent=[
            np.min(_phase_mean),
            np.max(_phase_mean),
            highfreq_range[0],
            highfreq_range[1],
        ],
        aspect="auto",
        cmap="inferno",
        origin="lower",
        interpolation="spline16",
    )
    if not phase_in_degrees:
        xticks_values = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        xticks_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
        plt.xticks(xticks_values, xticks_labels)
    plt.xlabel(f"{lowfreq_name} phase ({angle_type})")
    plt.ylabel(f"{highfreq_name} frequency (Hz)")
    plt.colorbar(label=f"{highfreq_name} amplitude (a.u.)", format="")
    show_or_save("phase_freq.png", save_folder=save_folder, save_plot=save_plots)


def filter_population_signal(
    spike_trains_A,
    spike_trains_B,
    layer_A,
    layer_B,
    input_type,
    num_steps_to_plot=1000,
    time_axis_first=True,
    delta_freq_range=[0.5, 4],
    theta_freq_range=[4, 8],
    alpha_freq_range=[8, 13],
    beta_freq_range=[13, 30],
    lowgamma_freq_range=[30, 80],
    highgamma_freq_range=[80, 150],
    sampling_rate=1000,
    save_plots=False,
    save_folder=".",
):
    # Reshape as (num_steps, num_neurons) if necessary
    if not time_axis_first:
        spike_trains_A = spike_trains_A.transpose()
        spike_trains_B = spike_trains_B.transpose()

    # Convert spike trains to population signal by summing and normalizing
    population_signal_A = spike_trains_A.mean(axis=1)
    population_signal_A = normalize_signal(population_signal_A)
    population_signal_B = spike_trains_B.mean(axis=1)
    population_signal_B = normalize_signal(population_signal_B)

    # Bands
    frequency_bands = [
        delta_freq_range,
        theta_freq_range,
        alpha_freq_range,
        beta_freq_range,
        lowgamma_freq_range,
        highgamma_freq_range,
    ]
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Low-Gamma", "High-Gamma"]

    # Discard last band if it extends beyond Nyquist freq
    if sampling_rate // 2 < 150:
        frequency_bands = frequency_bands[:-1]
        band_names = band_names[:-1]

    if layer_A == 0:
        layer_A_name = "Nerve"
    else:
        layer_A_name = f"Layer {layer_A}"

    if layer_B == 0:
        layer_B_name = "Nerve"
    else:
        layer_B_name = f"Layer {layer_B}"

    # Plots
    _, axes = plt.subplots(nrows=len(frequency_bands), ncols=1)
    for i, ax in enumerate(axes.ravel()):
        filtered_signal_A = bandpass_filter(
            signal=population_signal_A,
            freq_range=frequency_bands[i],
            sampling_rate=sampling_rate,
        )
        filtered_signal_B = bandpass_filter(
            signal=population_signal_B,
            freq_range=frequency_bands[i],
            sampling_rate=sampling_rate,
        )
        ax.plot(filtered_signal_A[:num_steps_to_plot], label=layer_A_name, lw=3)
        ax.plot(filtered_signal_B[:num_steps_to_plot], label=layer_B_name, lw=3)
        ax.set_xticks([])
        ax.set_yticks([])
        signal_mean = filtered_signal_A.mean()
        signal_std = filtered_signal_A.std()
        ax.set_ylim(signal_mean - 4 * signal_std, signal_mean + 4 * signal_std)
        ax.set_title(
            f"{band_names[i]} ({frequency_bands[i][0]}" f"-{frequency_bands[i][1]} Hz)"
        )
    ax.legend(loc="upper right")
    plt.tight_layout()
    show_or_save(
        f"filtered_population_signal_{input_type}.png",
        save_folder=save_folder,
        save_plot=save_plots,
    )


def pac_analysis(
    all_spikes,
    waveforms,
    wav_lens,
    num_utterances=2,
    num_phase_bins=18,
    num_surrogates=1000,
    sampling_rate_spikes=1000,
    sampling_rate_waveform=16000,
    phase_in_degrees=False,
):
    """
    Compute significance of phase-amplitude coupling both intra and inter-layer
    for all frequency bands. The input all_spikes must be a list of arrays each
    with shape (batch, time, feats) representing the spike trains from a layer.
    The analysis is done separately on num_utterances utterances and PAC is
    considered significant only if a small enough permutation testing p-value
    is measured for both MVL and MI metrics on all utterances.
    """

    if num_utterances >= all_spikes[0].shape[0]:
        raise ValueError(
            "Number of utterances should be smaller than batch size - 1 as "
            "last channel represents noise (not an utterance)."
        )

    # Define possible intra and interlayer combinations (including input waveform)
    num_layers = len(all_spikes)
    layer_combinations = [
        (i, j) for i in range(num_layers + 1) for j in range(i, num_layers + 1)
    ]
    layer_combinations.append("all")
    num_layer_combinations = len(layer_combinations)

    # Define frequency bands
    lowfreq_names = ["Delta", "Theta", "Alpha", "Beta"]
    highfreq_names = ["Low-Gamma", "High-Gamma"]
    lowfreq_ranges = [0.5, 4, 8, 13, 30]
    highfreq_ranges = [30, 80, 150]
    num_lowfreq = len(lowfreq_names)

    # Discard last band if it extends beyond Nyquist freq
    if sampling_rate_spikes // 2 < 150:
        highfreq_ranges = highfreq_ranges[:-1]
        highfreq_names = highfreq_names[:-1]
    num_highfreq = len(highfreq_names)

    # Initialize outputs
    pvalues_mvl = np.zeros(
        (num_utterances, num_layer_combinations, num_lowfreq, num_highfreq)
    )
    pvalues_mi = np.zeros(
        (num_utterances, num_layer_combinations, num_lowfreq, num_highfreq)
    )

    # Initialize progress bar
    total_iterations = (
        num_utterances * num_lowfreq * num_highfreq * num_layer_combinations
    )
    progress_bar = tqdm(total=total_iterations, desc="Progress", unit="iteration")

    # Loop over all utterances
    for utt_id in range(num_utterances):
        utt_spikes_steps = int(all_spikes[0].shape[1] * wav_lens[utt_id])
        utt_waveform_steps = int(waveforms[0].shape[0] * wav_lens[utt_id])
        utt_spikes = [
            all_spikes[i][utt_id, :utt_spikes_steps] for i in range(num_layers)
        ]
        utt_waveform = waveforms[utt_id, :utt_waveform_steps]

        # Loop over lowfreq-highfreq combinations for PAC
        for i, lowfreq_name in enumerate(lowfreq_names):
            lowfreq_range = [lowfreq_ranges[i], lowfreq_ranges[i + 1]]

            for j, highfreq_name in enumerate(highfreq_names):
                highfreq_range = [highfreq_ranges[j], highfreq_ranges[j + 1]]

                # Loop over intralayer, interlayer, whole network and input waveform
                for k, layer_combination in enumerate(layer_combinations):

                    if layer_combination == "all":
                        lowfreq_signal = np.concatenate(utt_spikes, axis=1)
                        highfreq_signal = lowfreq_signal.copy()
                        lowfreq_signal_type, highfreq_signal_type = "spikes", "spikes"
                    elif layer_combination == (0, 0):
                        lowfreq_signal = utt_waveform
                        highfreq_signal = lowfreq_signal.copy()
                        lowfreq_signal_type, highfreq_signal_type = (
                            "waveform",
                            "waveform",
                        )
                    elif layer_combination[0] == 0:
                        lowfreq_signal = utt_waveform
                        highfreq_signal = utt_spikes[layer_combination[1] - 1]
                        lowfreq_signal_type, highfreq_signal_type = "waveform", "spikes"
                    else:
                        lowfreq_signal = utt_spikes[layer_combination[0] - 1]
                        highfreq_signal = utt_spikes[layer_combination[1] - 1]
                        lowfreq_signal_type, highfreq_signal_type = "spikes", "spikes"

                    # Compute observed and surrogate metrics
                    (mvl_obs, mi_obs, mvl_surr, mi_surr, _, _, _, _, _, _) = (
                        compute_pac(
                            lowfreq_signal=lowfreq_signal,
                            highfreq_signal=highfreq_signal,
                            lowfreq_signal_type=lowfreq_signal_type,
                            highfreq_signal_type=highfreq_signal_type,
                            lowfreq_range=lowfreq_range,
                            highfreq_range=highfreq_range,
                            num_phase_bins=num_phase_bins,
                            num_surrogates=num_surrogates,
                            sampling_rate_spikes=sampling_rate_spikes,
                            sampling_rate_waveform=sampling_rate_waveform,
                            time_axis_first=True,
                            phase_in_degrees=phase_in_degrees,
                        )
                    )

                    # Evaluate p-value of observation
                    pvalues_mvl[utt_id, k, i, j] = compute_pvalue(mvl_surr, mvl_obs)
                    pvalues_mi[utt_id, k, i, j] = compute_pvalue(mi_surr, mi_obs)

                    # Update the progress bar
                    progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    return (pvalues_mvl, pvalues_mi, layer_combinations, lowfreq_names, highfreq_names)


if __name__ == "__main__":
    """
    This is just to test the functionality of the defined functions
    using some randomly generated spike trains.
    """

    # Options
    batch_size, num_steps, num_neurons = 8, 4000, 128
    sampling_rate_spikes = 1000
    sampling_rate_waveform = 16000
    num_phase_bins = 18
    num_layers = 4
    num_surrogates = 1000
    utterance_id = 0
    wav_lens = batch_size * [1.0]
    plot_rates = True
    plot_pac = True
    plot_signal = True
    run_pac_analysis = False

    # Fixed parameters
    dt = 1 / sampling_rate_spikes
    duration = num_steps * dt
    time_bins = np.arange(0, duration, dt)
    input_type = "noise"
    num_steps_waveform = int(duration * sampling_rate_waveform)

    # Generate random spike trains
    thres = np.random.uniform(size=(1, 1, num_neurons), low=0.9, high=0.995)
    all_spikes = [
        (np.random.uniform(size=(batch_size, num_steps, num_neurons)) > thres).astype(
            np.float32
        )
        for _ in range(num_layers)
    ]
    waveforms = np.zeros((batch_size, num_steps_waveform))
    for i in range(batch_size):
        waveforms[i, :] = normalize_signal(np.random.uniform(size=(num_steps_waveform)))

    _spikes = []
    for i in range(num_layers):
        _spikes.append(all_spikes[i][utterance_id].transpose())

    if plot_rates:
        firing_rates(
            all_spikes=all_spikes,
            duration=duration,
            num_steps=num_steps,
            input_type=input_type,
            bin_width=4,
            utterance_id=utterance_id,
            sampling_rate=sampling_rate_spikes,
            use_single_utterance=True,
            save_plot=False,
        )
    if plot_pac:
        pac_plots(
            lowfreq_signal=_spikes[0],
            highfreq_signal=_spikes[num_layers - 1],
            lowfreq_signal_type="spikes",
            highfreq_signal_type="spikes",
            lowfreq_range=[3, 8],
            highfreq_range=[30, 80],
            lowfreq_name="Theta",
            highfreq_name="Low-Gamma",
            sampling_rate_spikes=sampling_rate_spikes,
            sampling_rate_waveform=sampling_rate_waveform,
            num_phase_bins=num_phase_bins,
            num_surrogates=10000,
            time_axis_first=False,
            phase_in_degrees=False,
            save_plots=False,
        )
    if plot_signal:
        filter_population_signal(
            spike_trains_A=_spikes[0],
            spike_trains_B=_spikes[num_layers - 1],
            layer_A=0,
            layer_B=num_layers - 1,
            input_type="spikes",
            num_steps_to_plot=1000,
            time_axis_first=False,
            delta_freq_range=[0.5, 4],
            theta_freq_range=[4, 8],
            alpha_freq_range=[8, 13],
            beta_freq_range=[13, 30],
            lowgamma_freq_range=[30, 80],
            highgamma_freq_range=[80, 150],
            sampling_rate=sampling_rate_spikes,
            save_plots=False,
        )
    if run_pac_analysis:
        (pvalues_mvl, pvalues_mi, layer_combinations, lowfreq_names, highfreq_names) = (
            pac_analysis(
                all_spikes=all_spikes,
                waveforms=waveforms,
                wav_lens=wav_lens,
                num_utterances=2,
                num_phase_bins=num_phase_bins,
                num_surrogates=num_surrogates,
                sampling_rate_spikes=sampling_rate_spikes,
                sampling_rate_waveform=sampling_rate_waveform,
                phase_in_degrees=False,
            )
        )
        write_significance_mvl_mi(
            pvalues_mvl,
            pvalues_mi,
            layer_combinations,
            lowfreq_names,
            highfreq_names,
            save_folder=".",
        )

    print("\nDone.\n")
