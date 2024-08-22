#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""
Speechbrain recipe for training a phoneme recognizer on TIMIT,
based on speechbrain/recipes/TIMIT/ASR/CTC/train.py and adapted
for our SNN-based ASR encoder with firing rate regularizers.

To run this recipe, do the following:
> python run_training.py configs/timit_train.yaml
"""

import logging
import sys

import speechbrain as sb
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import if_main_process
from speechbrain.utils.distributed import run_on_main
from timit_prepare import prepare_timit
from train import dataio_prep

logger = logging.getLogger(__name__)


class ASR_Brain(sb.Brain):
    """
    Trainer class for speech recognition.
    """

    def compute_forward(self, batch, stage):

        # Get elements from input batch
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Apply data augmentation to waveform
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        # Compute acoustic features
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # Pass features through encoder
        out, all_spikes = self.modules.model(feats)

        # Compute firing rate tensor with shape (layers, neurons)
        duration = all_spikes[0].shape[1] * hparams["hop_length"] * 1e-3
        self.firing_rates = [
            torch.mean(x.sum(dim=1) / duration, dim=0) for x in all_spikes
        ]
        self.firing_rates = torch.cat(self.firing_rates, dim=-1)

        # Phoneme classifier from encoder outputs
        out = self.modules.classifier(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):

        # Get model predictions and ground truths
        pout, pout_lens = predictions
        phns, phn_lens = batch.phn_encoded

        # Waveform augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)

        # Compute ctc loss
        ctc_loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(batch.id, pout, phns, pout_lens, phn_lens)
        loss = ctc_loss

        # Compute regularizer loss
        if self.hparams.regularize_firing_rate:
            reg_quiet = F.relu(self.hparams.reg_fmin_hz - self.firing_rates).mean()
            reg_burst = F.relu(self.firing_rates - self.hparams.reg_fmax_hz).mean()
            reg_loss = self.hparams.reg_factor * (reg_quiet + reg_burst)
            loss += reg_loss

        # Decode probabilities into phonemes
        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per},
                min_keys=["PER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    w.write("CTC loss stats:\n")
                    self.ctc_metrics.write_stats(w)
                    w.write("\nPER stats:\n")
                    self.per_metrics.write_stats(w)
                    print(
                        "CTC and PER stats written to ",
                        self.hparams.test_wer_file,
                    )


# Begin Recipe!
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

    # Create csv file for noise augmentations
    run_on_main(hparams["prepare_noise_data"])

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

    # Training and validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
