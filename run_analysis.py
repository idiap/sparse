#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
Speechbrain recipe for analysing a phoneme recognizer on TIMIT
test split, based on speechbrain/recipes/TIMIT/ASR/CTC/train.py
and adapted to analyse spike trains across SNN-based ASR encoder.

To run this recipe, do the following:
> python run_analysis.py configs/timit_analysis.yaml
"""

import logging
import os
import sys

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from timit_prepare import prepare_timit

from sparse.analysis import analyse_spikes

logger = logging.getLogger(__name__)


class ASR_Brain(sb.Brain):
    """
    Trainer class for speech recognition.
    """

    def compute_forward(self, batch, stage):

        # Get elements from input batch
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Compute acoustic features
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # Add Gaussian noise as last batch element for analysing response to noise
        noise = torch.randn(1, feats.shape[1], feats.shape[2]).to(feats.device)
        noise = self.modules.normalize(noise, torch.ones(1))
        feats = torch.cat((feats, noise), dim=0)
        wav_lens = torch.cat((wav_lens, torch.ones(1).to(feats.device)), dim=0)

        # Pass waveform and feats through encoder
        out, all_spikes = self.modules.model(feats)

        # Run analysis of spikes produced across encoder
        analyse_spikes(
            model=self.modules.model,
            all_spikes=all_spikes,
            feats=feats,
            wavs=wavs,
            wav_lens=wav_lens,
            hop_length_ms=self.hparams.hop_length,
            phase_layer=self.hparams.phase_layer,
            amplitude_layer=self.hparams.amplitude_layer,
            phase_freq_range=self.hparams.phase_freq_range,
            amplitude_freq_range=self.hparams.amplitude_freq_range,
            phase_band_name=self.hparams.phase_band_name,
            amplitude_band_name=self.hparams.amplitude_band_name,
            num_utterances_pac_analysis=self.hparams.num_utterances_pac_analysis,
            utterance_id=self.hparams.utterance_id,
            plot_parameters=self.hparams.plot_parameters,
            plot_weights=self.hparams.plot_weights,
            plot_spikes=self.hparams.plot_spikes,
            plot_rates=self.hparams.plot_rates,
            plot_phase_amplitude_coupling=self.hparams.plot_pac,
            plot_filtered_population_signal=self.hparams.plot_filtered_population,
            run_pac_analysis=self.hparams.run_pac_analysis,
            save_plots=self.hparams.save_plots,
            save_folder=self.hparams.plots_folder,
        )

        # Exit after analysis is complete
        sys.exit("\n\nAnalysis completed. Exiting the script.\n\n")


def dataio_prep(hparams):
    """
    Creates the datasets and their data processing pipelines.
    Here test data is sorted by descending duration so that we analyse the
    longest sequences and not the shortest.
    """

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration", reverse=True)

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-gpu dpp support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":

    # Load config
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing TIMIT and annotation into csv files)
    from timit_prepare import prepare_timit  # noqa

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create json files for dataset
    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "skip_prep": hparams["skip_prep"],
            "uppercase": hparams["uppercase"],
        },
    )

    # Dataset IO preparation
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # Testing (triggers the analysis)
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
