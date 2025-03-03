import sys
path_to_add = '../'
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

import pdb
import os
# import readline
# import gnureadline
os.environ['TERM'] = 'xterm'

BEAM_SIZE = 1
from psy import Masker
import argparse

import torch
import torchaudio
from pathlib import Path
from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter
from seamless_communication.cli.expressivity.predict.pretssel_generator import (
    PretsselGenerator, )
from seamless_communication.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)
from seamless_communication.inference import Translator, SequenceGeneratorOptions

from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.store import add_gated_assets
from fairseq2.generation import NGramRepeatBlockProcessor
from typing import List, Optional, Tuple, Union, cast
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from seamless_communication.inference.generator import (
    SequenceGeneratorOptions,
    UnitYGenerator,
)
from fairseq2.nn.incremental_state import IncrementalStateBag
# from fairseq2.generation.generator import Hypothesis
from dataclasses import dataclass
from torch import Tensor

torch.autograd.set_detect_anomaly(True)
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


@dataclass
class Hypothesis:
    """Represents a hypothesis produced by a sequence generator."""

    seq: Tensor
    """The generated sequence. *Shape:* :math:`(S)`, where :math:`S` is the
    sequence length."""

    score: Optional[Tensor]
    """The score of the hypothesis. *Shape:* Scalar."""

    step_scores: Optional[Tensor]
    """The score of each sequence step. *Shape:* :math:`(S)`, where :math:`S` is
    the sequence length."""




def remove_prosody_tokens_from_text(text: str) -> str:
    # filter out prosody tokens, there is only emphasis '*', and pause '='
    text = text.replace("*", "").replace("=", "")
    text = " ".join(text.split())
    return text


def set_generation_opts(args):
    # Set text, unit generation opts.
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=args.text_generation_beam_size,
        soft_max_seq_len=(
            args.text_generation_max_len_a,
            args.text_generation_max_len_b,
        ),
    )
    if args.text_unk_blocking:
        text_generation_opts.unk_penalty = torch.inf
    if args.text_generation_ngram_blocking:
        text_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size)

    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=args.unit_generation_beam_size,
        soft_max_seq_len=(
            args.unit_generation_max_len_a,
            args.unit_generation_max_len_b,
        ),
    )
    if args.unit_generation_ngram_blocking:
        unit_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size)
    return text_generation_opts, unit_generation_opts


class Seamless_configuration:
    def __init__(self):
        self.task = "s2tt"
        self.src_lang = None
        self.output_path = None
        self.text_generation_beam_size = BEAM_SIZE
        self.text_generation_max_len_a = 1
        self.text_generation_max_len_b = 200
        self.text_generation_ngram_blocking = False
        self.no_repeat_ngram_size = 4
        self.unit_generation_beam_size = BEAM_SIZE
        self.unit_generation_max_len_a = 25
        self.unit_generation_max_len_b = 50
        self.unit_generation_ngram_blocking = False
        self.unit_generation_ngram_filtering = False
        self.text_unk_blocking = False
        self.duration_factor = 1.0
        self.tgt_lang = None


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

import math
import torch
import numpy as np
import torch.nn.functional as F
import random
import string
import time
# import os
import argparse
import scipy.io.wavfile as wav
import librosa
import soundfile
# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

# Sampling rate of the input files
Fs = 16000
UN_LEN = 3


from dataset import wav_dataset_librosa_cut as my_dataset
from torch.utils.data import DataLoader


def ctc_beam_search_decoder():
    return 0


masker = Masker(device=device)

import math
import torch.nn as nn


class AMSoftmaxLoss(torch.nn.Module):

    def __init__(self, s=30.0, m=0.35):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        # Ensure logits and labels have the correct shape
        assert logits.dim() == 2, "Logits should be 2-dimensional"
        assert labels.dim() == 1, "Labels should be 1-dimensional"
        assert logits.size(0) == labels.size(
            0), "Logits and labels should have the same batch size"

        # Calculate the cosine similarity
        cosine = logits

        # Create the target logit with margin
        phi = cosine - self.m

        # Create a mask to select the target logit
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply the margin to the target logit
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the logits
        output *= self.s

        # Calculate the cross-entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class SharpnessLoss(nn.Module):

    def __init__(self, alpha=1.0, reduction='mean'):
        super(SharpnessLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # Cross Entropy Loss
        # pdb.set_trace()
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)

        # Compute the logits for the target classes
        target_logits = logits.gather(1, targets.view(-1, 1)).squeeze()

        # Compute the sharpness penalty (negative target logits to maximize them)
        sharpness_penalty = -target_logits

        # pdb.set_trace()
        # Combine the losses
        loss = ce_loss + self.alpha * sharpness_penalty.mean()

        return loss


ce_loss_func = SharpnessLoss(alpha=0.5)

from typing import Sequence
from torch.nn.functional import log_softmax


@dataclass
class BeamStep:
    """Represents the output of a beam search algorithm."""

    seq_indices: Tensor
    """The beam sequence indices. *Shape:* :math:`(B)`, where :math:`B` is the
    beam size."""

    vocab_indices: Tensor
    """The vocabulary indices. *Shape:* Same as ``seq_indices``."""

    scores: Tensor
    """The scores. *Shape:* Same as ``seq_indices``."""

    def masked_select(self, mask: Tensor):
        """Reduce the beam to the sequences included in ``mask``."""
        seq_indices = self.seq_indices.masked_select(mask)

        vocab_indices = self.vocab_indices.masked_select(mask)

        scores = self.scores.masked_select(mask)

        return BeamStep(seq_indices, vocab_indices, scores)

    def first(self, count: int):
        """Slice the beam to the first ``count`` sequences."""
        seq_indices = self.seq_indices[:count]

        vocab_indices = self.vocab_indices[:count]

        scores = self.scores[:count]

        return BeamStep(seq_indices, vocab_indices, scores)

    @staticmethod
    def merge(steps):
        """Merge ``steps`` into a single beam."""
        seq_indices = torch.cat([s.seq_indices for s in steps])

        vocab_indices = torch.cat([s.vocab_indices for s in steps])

        scores = torch.cat([s.scores for s in steps])

        return BeamStep(seq_indices, vocab_indices, scores)


@dataclass
class Seq2SeqGeneratorOutput:
    hypotheses: List[List]
    """The list of hypothesis generated per prompt, ordered by score."""

    encoder_output: Tensor
    """The encoder output used in encoder-decoder attention. *Shape:*
    :math:`(N,S_{enc},M)`, where :math:`N` is the batch size, :math:`S_{enc}` is
    the encoder output sequence length, and :math:`M` is the dimensionality of
    the model."""

    encoder_padding_mask: Optional[PaddingMask]
    """The padding mask of :attr:`encoder_output`. *Shape:* :math:`(N,S_{enc})`,
    where :math:`N` is the batch size and :math:`S_{enc}` is the encoder output
    sequence length."""


def my_fbank_extracter(waveform,
                       sample_rate,
                       num_mel_bins=80,
                       waveform_scale=2**15,
                       channel_last=True,
                       standardize=False,
                       device='cpu',
                       dtype=dtype):
    # 缩放波形num_mel_bins=80,
    waveform = waveform * waveform_scale

    # 转换通道维度
    if channel_last:
        waveform = waveform.transpose(0, 1)
    fbank = torchaudio.compliance.kaldi.fbank(waveform,
                                              blackman_coeff=0.42,
                                              channel=-1,
                                              dither=0.0,
                                              energy_floor=1.0,
                                              frame_length=25.0,
                                              frame_shift=10.0,
                                              high_freq=0.0,
                                              htk_compat=False,
                                              low_freq=20.0,
                                              min_duration=0.0,
                                              num_mel_bins=num_mel_bins,
                                              preemphasis_coefficient=0.97,
                                              raw_energy=True,
                                              remove_dc_offset=True,
                                              round_to_power_of_two=True,
                                              sample_frequency=16000.0,
                                              snip_edges=True,
                                              subtract_mean=False,
                                              use_energy=False,
                                              use_log_fbank=True,
                                              use_power=True,
                                              vtln_high=-500.0,
                                              vtln_low=100.0,
                                              vtln_warp=1.0,
                                              window_type='povey')
    return fbank


my_ctc_loss_fn = torch.nn.CTCLoss()


class get_logits():

    def __init__(self, wav, phrase):
        """
        Compute the logits for a given waveform.
        """
        wav = wav.transpose(0, 1)

        my_fbank = my_fbank_extracter(wav, 16000).squeeze(0)
        fbank = my_fbank.to(dtype)

        std, mean = torch.std_mean(fbank, dim=0)
        fbank = fbank.subtract(mean).divide(std)

        src = SequenceData(
            seqs=fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([fbank.shape[0]]),
            is_ragged=False,
        )

        text_generation_opts, unit_generation_opts = set_generation_opts(
            s_config)


        target_phase_list = []
        self.text_phase_list = []
        if use_target_lang[0] != "eng":
            start_index = 0
        else:
            start_index = 1
            target_phase_now = use_token_encoder[0](phrase).to(device)
            target_phase_list.append(target_phase_now.unsqueeze(0))
            self.text_phase_list.append(phrase)
        for now_tl_index in range(start_index, len(use_target_lang)):
            text_output_now, _ = text_translator.predict(
                str(phrase),
                "t2tt",
                use_target_lang[now_tl_index],
                src_lang=s_config.tgt_lang,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                unit_generation_ngram_filtering=s_config.
                unit_generation_ngram_filtering,
            )
            target_phase_now = use_token_encoder[now_tl_index](str(
                text_output_now[0])).to(device)
            self.text_phase_list.append(str(text_output_now[0]))
            target_phase_list.append(target_phase_now.unsqueeze(0))

        src = cast(SequenceData, src)

        acoustic_seqs, padding_mask = get_seqs_and_padding_mask(src)
        self.generator = UnitYGenerator(
            translator.model,
            translator.text_tokenizer,
            s_config.tgt_lang,
            # unit_tokenizer if output_modality == Modality.SPEECH else None,
            None,
            text_opts=text_generation_opts,
            unit_opts=unit_generation_opts,
        )

        target_prefix_seq = self.generator.s2t_converter.target_prefix_seq
        self.target_prefix_seqs = target_prefix_seq.expand(
            acoustic_seqs.shape[0], -1)
        min_prompt_len, min_prompt_idx = self.target_prefix_seqs.size(
            1), 0  # 2,0
        # prefill_len = min_prompt_len #
        self.max_prompt_len, max_prompt_idx = self.target_prefix_seqs.size(
            1), 0
        max_gen_len = text_generation_opts.soft_max_seq_len
        a_term, b_term = max_gen_len
        max_source_len = acoustic_seqs.size(1)
        max_gen_len = int(a_term * max_source_len + b_term)
        max_seq_len = 1024
        self.max_seq_len = min(max_seq_len, self.max_prompt_len + max_gen_len)
        self.num_prompts = self.target_prefix_seqs.size(0)
        self.all_logits = torch.randn(target_phase_list[0].shape[1], 1, 256102)

        self.targets = torch.cat((torch.tensor([
            self.generator.s2t_converter.generator.model.target_vocab_info.
            eos_idx
        ]).unsqueeze(0).to(device), target_phase_list[0]),
                                 dim=1)
        self.targets_more_list = []
        for now_tl_index in range(1, len(use_target_lang)):
            self.targets_more_list.append(\
                torch.cat((torch.tensor([self.generator.s2t_converter.generator.model.target_vocab_info.eos_idx]).unsqueeze(0).to(device), target_phase_list[now_tl_index]), dim=1))

        self.min_prompt_len = min_prompt_len
        self.prefill_len = min_prompt_len  #
        min_gen_len = 1
        self.min_seq_len = min(max_seq_len, self.max_prompt_len + min_gen_len)

    def logits(self, wav, teacher_forcing):
        """
        Compute the logits for a given waveform.
        """
        wav = wav.transpose(0, 1)

        my_fbank = my_fbank_extracter(wav, 16000).squeeze(0)
        fbank = my_fbank.to(dtype)
        std, mean = torch.std_mean(fbank, dim=0)
        fbank = fbank.subtract(mean).divide(std)

        src = SequenceData(
            seqs=fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([fbank.shape[0]]),
            is_ragged=False,
        )

        src = cast(SequenceData, src)
        acoustic_seqs, padding_mask = get_seqs_and_padding_mask(src)

        encoder_output, encoder_padding_mask = self.generator.s2t_converter.generator.model.encode(
            acoustic_seqs, padding_mask)

        ce_loss = 0
        texts = ""
        if teacher_forcing:
            ######################################################################### teacher forcing
            texts_token = []
            temp_state_bag = IncrementalStateBag(self.max_seq_len)
            temp_seqs = self.targets
            for j in range(1, temp_seqs.shape[1]):
                # temp_state_bag = IncrementalStateBag(max_seq_len)
                decoder_output, decoder_padding_mask = self.generator.s2t_converter.generator.model.decode(
                    temp_seqs[:, j - 1:j],
                    None,  # We never use PAD in incremental decoding.
                    encoder_output,
                    encoder_padding_mask,
                    state_bag=temp_state_bag,
                )
                model_output = self.generator.s2t_converter.generator.model.project(
                    decoder_output, decoder_padding_mask)
                temp_state_bag.increment_step_nr()
                logits = model_output.logits  # torch.Size([1, 1, 256102])
                lprobs = logits
                if j > 1:
                    ce_loss = ce_loss + ce_loss_func(
                        lprobs.squeeze(1), temp_seqs[:, j])
                top_scores, top_indices = torch.topk(lprobs, k=1)
                texts_token.append(top_indices)
                # self.all_logits[j-1] = logits
                temp_state_bag.reorder(
                    (top_indices // lprobs.shape[-1]).squeeze(0).squeeze(0))

            texts = [
                self.generator.s2t_converter.text_decoder(
                    torch.cat(texts_token, dim=1).squeeze().squeeze())
            ]
            return_texts_token = torch.cat(texts_token,
                                           dim=1).squeeze().squeeze()

            ##### more lang
            for temp_seqs in self.targets_more_list:
                texts_token = []
                # for i in range(self.targets_cmn.shape[1], self.targets_cmn.shape[1]+1):
                temp_state_bag = IncrementalStateBag(self.max_seq_len)
                # temp_seqs = self.targets_cmn[:, :i]
                for j in range(1, temp_seqs.shape[1]):
                    # temp_state_bag = IncrementalStateBag(max_seq_len)
                    decoder_output, decoder_padding_mask = self.generator.s2t_converter.generator.model.decode(
                        temp_seqs[:, j - 1:j],
                        None,  # We never use PAD in incremental decoding.
                        encoder_output,
                        encoder_padding_mask,
                        state_bag=temp_state_bag,
                    )
                    model_output = self.generator.s2t_converter.generator.model.project(
                        decoder_output, decoder_padding_mask)
                    temp_state_bag.increment_step_nr()
                    logits = model_output.logits  # torch.Size([1, 1, 256102])
                    lprobs = logits
                    if j > 1:
                        ce_loss = ce_loss + ce_loss_func(
                            lprobs.squeeze(1), temp_seqs[:, j])
                    top_scores, top_indices = torch.topk(lprobs, k=1)
                    texts_token.append(top_indices)
                    # self.all_logits[j-1] = logits
                    temp_state_bag.reorder(
                        (top_indices //
                         lprobs.shape[-1]).squeeze(0).squeeze(0))

                texts.append(
                    self.generator.s2t_converter.text_decoder(
                        torch.cat(texts_token, dim=1).squeeze().squeeze()))
        
        logging.info("\n" + "*" * 40)
        return self.all_logits, texts, ce_loss, return_texts_token




def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    max_label_length = max(label_lengths)
    label_sparse_list = []

    for i in range(batch_size):
        label_sparse = labels[i, :label_lengths[i]]
        label_sparse_list.append(label_sparse)

    label_sparse_tensor = torch.cat(label_sparse_list)
    label_indices = torch.nonzero(label_sparse_tensor >= 0)
    vals_sparse = label_sparse_tensor[label_indices[:, 0], label_indices[:, 1]]

    return torch.sparse_coo_tensor(label_indices.t(), vals_sparse, size=(batch_size, max_label_length))


def kl_divergence(latents):
    mean, logvar = torch.mean(latents), torch.log(torch.var(latents, unbiased=False) + 1e-8)
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl


def add_tensors_with_padding(tensor_a, tensor_b):
    size_a = tensor_a.shape[1]
    size_b = tensor_b.shape[1]

    if size_a < size_b:
        pad_size = size_b - size_a
        tensor_a_padded = F.pad(tensor_a, (0, pad_size))
        result = tensor_a_padded + tensor_b
    elif size_b < size_a:
        pad_size = size_a - size_b
        tensor_b_padded = F.pad(tensor_b, (0, pad_size))
        result = tensor_a + tensor_b_padded
    else:
        result = tensor_a + tensor_b

    return result


class Attack:
    def __init__(self, impulse, phrase, freq_min, freq_max, batch_size, learning_rate, weight_decay, outdir, bp, psy):
        audio = torch.randn(Fs*UN_LEN).numpy()*0.005
        self.outdir = outdir
        assert len(audio.shape) == 1
        self.phrase = phrase
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.impulse_size = 1
        self.batch_size = batch_size

        self.original = torch.randn(1, Fs*UN_LEN, device=device)*0.005
        
        self.impulse = torch.tensor(impulse, dtype=torch.float32).to(device)
        
        from mustango import Mustango
        self.tango = Mustango("declare-lab/mustango", device)
        self.tango.vae.requires_grad_(False)
        self.tango.model.unet.requires_grad_(False)
        self.tango.model.text_encoder.encoder.requires_grad_(False)
        self.tango.model.text_encoder.shared.requires_grad_(True)
        self.tango.model.PE.global_time_embedding.angles.requires_grad_(True)
        self.tango.model.FME.translation_bias.requires_grad_(True)
        self.tango.model.FME.angles.requires_grad_(True)
        
        self.initial_prompt = "This techno song features a synth lead playing the main melody. This is accompanied by programmed percussion playing a simple kick focused beat. The hi-hat is accented in an open position on the 3-and count of every bar. The synth plays the bass part with a voicing that sounds like a cello. This techno song can be played in a club. The chord sequence is Gm, A7, Eb, Bb, C, F, Gm. The beat counts to 2. The tempo of this song is 256.0 beats per minute. The key of this song is G minor."
        
        num_channels_latents = self.tango.model.unet.config.in_channels
        inference_scheduler = self.tango.scheduler
        self.num_steps = 3
        inference_scheduler.set_timesteps(self.num_steps, device=device)
        self.latents = self.tango.model.prepare_latents(1, inference_scheduler, num_channels_latents, torch.float32, device).requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
                [self.latents] + \
                [self.tango.model.PE.global_time_embedding.angles] +\
                [self.tango.model.FME.translation_bias] + [self.tango.model.FME.angles],
                lr=learning_rate,
                weight_decay=weight_decay,
        )

        self.ctc_loss_func = torch.nn.CTCLoss()

        self.noise_ratio = torch.tensor(np.ones((1, ), dtype=np.float32), requires_grad=True, device=device, dtype=dtype)

        self.get_logits = get_logits(torch.randn(1,Fs*2), self.phrase)
        self.text_phase_list = self.get_logits.text_phase_list
        with torch.no_grad():
            self.beats, self.chords, self.chords_times = self.tango.music_model.generate(self.initial_prompt)
        # self.beats = [[[7.22, 7.6, 7.99, 8.36, 8.8, 9.14, 9.6, 0.2, 0.65, 1.14, 1.6, 2.02, 2.48, 2.97, 3.5, 3.99, 4.45, 4.95, 5.46, 5.93, 6.39, 6.82], \
        #                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]]
        # self.chords = ['G', 'D', 'Em', 'C']
        # self.chords_times = [0.46, 0.04, 0.09, 0.32]

    def run_step(self, imp_indices, itr, audio):
        # generate
        # [[[0.19, 0.63, 1.12, 1.56, 1.96, 2.41, 2.86, 3.37, 3.86, 4.32, 4.82, 5.26, 5.74, 6.16, 6.53, 6.97, 7.34, 7.72, 8.11, 8.51, 8.91, 9.28, 9.66, 9.96], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]]
        # ['Em', 'C', 'G', 'D']
        # [0.46, 2.04, 4.09, 6.32]
        self.original = audio.to(device)
        latents = self.tango.model.inference2(self.latents, [self.initial_prompt],
                                             self.beats,
                                            [self.chords],
                                            [self.chords_times],
                                            self.tango.scheduler, self.num_steps, 3.0)[:,:,:64,:]

        mel = self.tango.vae.decode_first_stage(latents)
        wave = self.tango.vae.decode_to_waveform(mel)
        
        final_input = add_tensors_with_padding(wave[0], self.original)
        
        self.conv_length = audio.shape[1]
        self.nfft = 2 ** int(math.ceil(math.log(self.conv_length, 2)))
        audio_fft = torch.fft.rfft(final_input, n=self.nfft)
        ir_fft = torch.fft.rfft(self.impulse[imp_indices].unsqueeze(0), n=self.nfft)
        convolved_fft = audio_fft * ir_fft
        ae_convolved = torch.fft.irfft(convolved_fft, n=self.nfft)[:,:self.conv_length]
        # Normalize the convolved audio
        max_audio = torch.max(torch.abs(ae_convolved), dim=1, keepdim=True).values
        ae_transformed = ae_convolved / max_audio * torch.max(torch.abs(final_input))
        small_noise = torch.randn(ae_transformed.size(), dtype=ae_transformed.dtype, device=ae_transformed.device)
        small_noise = small_noise * torch.tensor(1 - self.noise_ratio, dtype=ae_transformed.dtype, device=ae_transformed.device)
        final_input = torch.clamp((ae_transformed + small_noise)*(2 ** 15 - 1), min=-2 ** 15, max=2 ** 15 - 1) / (2 ** 15 - 1)
        
        logits, texts, ce_loss, return_texts_token = self.get_logits.logits(final_input, True)
        if (itr) % 10 == 0:
            soundfile.write(os.path.join(self.outdir, str(texts[0])+"_.wav"), final_input.squeeze().detach().cpu().numpy(), samplerate=Fs)

        psy_loss = kl_divergence(self.latents)
        
        sum_loss = ce_loss + psy_loss

        self.optimizer.zero_grad()
        sum_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latents, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.tango.model.PE.global_time_embedding.angles, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.tango.model.FME.translation_bias, max_norm=1.0)

        logging.info('Grad 1: %s' %(np.array_str(self.latents.grad[0][0][0].detach().cpu().numpy(), max_line_width=120)))
        logging.info('Grad 2: %s' %(np.array_str(self.tango.model.PE.global_time_embedding.angles.grad.detach().cpu().numpy(), max_line_width=120)))
        logging.info('Grad 3: %s' %(np.array_str(self.tango.model.FME.translation_bias.grad.detach().cpu().numpy(), max_line_width=120)))
        
        self.optimizer.step()

        decoded = [str(te) for te in texts]
        return decoded, logits, ce_loss, final_input, final_input, wave[0], return_texts_token


    def attack(self, outdir, num_iterations=5):
        # Create misc variables
        prefix = ''.join([random.choice(string.ascii_lowercase) for _ in range(3)])
        time_last, time_start = time.time(), time.time()

        # We'll make a bunch of iterations of gradient descent here
        itr = 0
        most_flag = False
        for itr_ in range(num_iterations + 1):
            if most_flag: break
            for audio in train_loader:
                itr += 1
                if itr > 12000:
                    most_flag = True
                    break
                indice = np.random.choice(self.impulse.shape[0])
                audio = audio["matrix"]
                # Reduce volume (multiply by a factor between 0 and 1)
                volume_factor = 0.01  # Adjust this value to control volume (0.3 = 30% of original volume)
                audio = audio * volume_factor
                # Actually do the optimization step
                decoded, logits, ce_loss, ae_transformed, ae_input, delta_filtered, return_texts_token = self.run_step(
                    indice, itr, audio)

                # Report progress
                ce_loss = [ce_loss]
                logging.info('Iter: %d, Elapsed Time: %.3f, Iter Time: %.3f\n\tLosses: %s\n\tDelta: %s' % \
                    (itr, time.time() - time_start, time.time() - time_last, ' '.join('% 6.2f' % x for x in ce_loss), np.array_str(delta_filtered.detach().cpu().numpy(), max_line_width=120)))
                time_last = time.time()

                # logging.info out some debug information every 5 iterations.
                if itr % 5 == 0:
                    logging.info('Recognition:\n\t' + '\n\t'.join(decoded))
                    logging.info(f"{return_texts_token[0]}")

                if set(decoded) == set(self.text_phase_list):
                    # Get the current constant
                    ratio = self.noise_ratio
                    logging.info('=> It: %d, Noise Ratio: %.3f' %
                                (itr, 1.0 - ratio[0]))

                    # Update with the new noise
                    self.noise_ratio = ratio * noise_step_ratio

                    logging.info(set(decoded))
                    if itr % 1000 == 0:
                        itr2 = itr + 1
                    else:
                        itr2 = itr
                    # wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr2)), Fs, np.array(np.clip(np.round(ae_input.squeeze(0).detach().cpu().numpy()*(2 ** 15 - 1)), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr2)), Fs, np.array(np.clip(np.round(delta_filtered.squeeze(0).detach().cpu().numpy()*(2 ** 15 - 1)), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    break

                if itr % 1000 == 0:
                    # wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr)), Fs, np.array(np.clip(np.round(ae_input.squeeze(0).detach().cpu().numpy()*(2 ** 15 - 1)), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                    wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr)), Fs, np.array(np.clip(np.round(delta_filtered.squeeze(0).detach().cpu().numpy()*(2 ** 15 - 1)), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))


def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest='input', required=False,
                        help='Input audio .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--imp', type=str, dest='impulse', nargs='+', required=True,
                        help='Input impulse response .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--target', type=str, required=True,
                        help='Target semantic')
    parser.add_argument('--out', type=str, required=True,
                        help='Directory for saving intermediate files')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='Batch size for generation')
    parser.add_argument('--freq_min', type=int, required=False, default=1000,
                        help='Lower limit of band pass filter for adversarial noise')
    parser.add_argument('--freq_max', type=int, required=False, default=4000,
                        help='Higher limit of band pass filter for adversarial noise')
    parser.add_argument('--lr', type=float, required=False, default=0.1,
                        help='Learning rate for optimization')
    parser.add_argument('--decay', type=float, required=False, default=0.001,
                        help='Weight decay for optimization')
    parser.add_argument('--iterations', type=int, required=False, default=5,
                        help='Maximum number of iterations of gradient descent')
    parser.add_argument('--session', type=str, required=False,
                        default=os.path.join(os.path.dirname(__file__), 'models/session_dump'),
                        help='Path for the session file taken from DeepSpeech')
    parser.add_argument('--bp', type=int, required=False, default=0,
                        help='if use bandpass')
    parser.add_argument('--psy', type=int, required=False, default=0,
                        help='if use psy')
    parser.add_argument('--noise', type=float, required=False, default=1.,
                        help='noise tep ratio')
    parser.add_argument('--tgtl', type=str, required=False, default="eng",
                        help='target language')
    parser.add_argument('--tgtm', type=str, required=False, default="seamlessM4T_large",
                        help='target model')
    parser.add_argument('--speech_pth', type=str, required=True, default="../../../core-code/LibriSpeech_wav/",
                        help='speech dataset path')
    args = parser.parse_args()



    global noise_step_ratio
    noise_step_ratio = args.noise
    global use_target_lang
    use_target_lang = args.tgtl.split(",")
    global use_token_encoder
    use_token_encoder = []


    global s_config
    global translator
    
    global train_loader
    train_set = my_dataset(args.speech_pth, flag="train", leng=Fs*2)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    s_config = Seamless_configuration()
    s_config.tgt_lang = "eng"

    logging.info(f"Running inference on {device=} with {dtype=}.")

    global text_translator
    text_translator = Translator(
            "seamlessM4T_large",
            vocoder_name_or_card=None,
            device=device,
            dtype=dtype,
        )
    translator = text_translator
    
    if "eng" in use_target_lang and use_target_lang[0] != "eng":
            use_target_lang.remove("eng")
            use_target_lang.insert(0, "eng")
    for lan in use_target_lang:
        use_token_encoder.append(
            translator.text_tokenizer.create_encoder(
                task="translation", lang=lan, mode="source", device=device)
        )

    logging.info(f'Command line: {args}')

    irs = []
    for i in range(len(args.impulse)):
        ir, fs = librosa.load(args.impulse[i], sr=Fs)
        assert fs == Fs
        irs.append(ir)

    # Pad the impulse responses
    maxlen = max(map(len, irs))
    for i in range(len(irs)):
        irs[i] = np.concatenate(
            (irs[i], np.zeros(maxlen - irs[i].shape[0], dtype=irs[i].dtype)))
    irs = np.array(irs)

    # Set up the attack class and run it
    attack = Attack(irs, args.target, freq_min=args.freq_min, freq_max=args.freq_max,
                    batch_size=args.batch_size, learning_rate=args.lr, weight_decay=args.decay, outdir=args.out, bp=args.bp, psy=args.psy)
    if not os.path.exists(args.out): os.makedirs(args.out)
    import re
    # max_b, max_b_file = None, None
    pattern = re.compile(r"([a-zA-Z]{3})-adv-(\d+)\.wav$")
    iter_list = []
    for filename in os.listdir(args.out):
        match = pattern.match(filename)
        if match:
            iter_list.append(filename)
    iter_list = sorted(iter_list, key=lambda name: int(name.split('-')[2].split('.')[0]))
    for name in iter_list:
        if int(name.split('-')[2].split('.')[0])%1000 != 0:
            logging.info(("#"*30+"\n")*3)
            logging.info(f"already end with {name}\n")
            return 0
        elif int(name.split('-')[2].split('.')[0]) == args.iterations:
            logging.info(("#"*30+"\n")*3)
            logging.info(f"already end with {name}\n")
            return 0
        elif int(name.split('-')[2].split('.')[0]) >= 12000:
            logging.info(("#"*30+"\n")*3)
            logging.info(f"already end with {name}\n")
            return 0
        
    with open(os.path.join(args.out, "target.txt"), "w") as f:
        f.write(args.target)
        f.close()
    attack.attack(outdir=args.out, num_iterations=args.iterations)


if __name__ == '__main__':
    main()
