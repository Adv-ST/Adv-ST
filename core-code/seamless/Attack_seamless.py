import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import os
os.environ['TERM'] = 'xterm'

from psy import Masker
import argparse

import torch
import torchaudio
from pathlib import Path

from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter

from seamless_communication.cli.expressivity.predict.pretssel_generator import (
    PretsselGenerator,
)
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
from dataclasses import dataclass
from torch import Tensor
torch.autograd.set_detect_anomaly(True)
BEAM_SIZE = 1

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
            ngram_size=args.no_repeat_ngram_size
        )

    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=args.unit_generation_beam_size,
        soft_max_seq_len=(
            args.unit_generation_max_len_a,
            args.unit_generation_max_len_b,
        ),
    )
    if args.unit_generation_ngram_blocking:
        unit_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )
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
import argparse
import scipy.io.wavfile as wav
import librosa
import soundfile
from collections import Counter
# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

# Sampling rate of the input files
Fs = 16000


def ctc_beam_search_decoder():
    return 0
masker = Masker(device = device)

ce_loss_func = torch.nn.CrossEntropyLoss()


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


def find_most_frequent_string(strings):
    if not strings:
        return None, 0
    counter = Counter(strings)
    most_common_string, most_common_count = counter.most_common(1)[0]
    return most_common_string, most_common_count



def my_fbank_extracter(waveform, sample_rate, num_mel_bins=80, waveform_scale=2**15, 
                      channel_last=True, standardize=False, device='cpu', dtype=dtype):
    # 缩放波形num_mel_bins=80,
    waveform = waveform * waveform_scale

    # 转换通道维度
    if channel_last:
        waveform = waveform.transpose(0, 1)
    fbank = torchaudio.compliance.kaldi.fbank(
                waveform, 
                blackman_coeff = 0.42, 
                channel =  -1, 
                dither= 0.0, 
                energy_floor = 1.0, 
                frame_length = 25.0, 
                frame_shift = 10.0, 
                high_freq = 0.0, 
                htk_compat = False, 
                low_freq = 20.0, 
                min_duration = 0.0, 
                num_mel_bins = num_mel_bins, 
                preemphasis_coefficient = 0.97, 
                raw_energy = True, 
                remove_dc_offset = True, 
                round_to_power_of_two = True, 
                sample_frequency = 16000.0, 
                snip_edges = True, 
                subtract_mean = False, 
                use_energy = False, 
                use_log_fbank = True, 
                use_power = True, 
                vtln_high = -500.0, 
                vtln_low = 100.0, 
                vtln_warp = 1.0, 
                window_type = 'povey')
    return fbank


my_ctc_loss_fn = torch.nn.CTCLoss()

class get_logits():
    def __init__(self, tgtm, cycle_flag, outdir, wav, phrase):
        """
        Compute the logits for a given waveform.
        """
        wav = wav.transpose(0, 1)
        my_fbank = my_fbank_extracter(wav, 16000).squeeze(0)
        fbank = my_fbank.to(dtype)
        if tgtm == "seamless_expressivity":
            gcmvn_fbank = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
            src_gcmvn = SequenceData(
                seqs=gcmvn_fbank.unsqueeze(0),
                seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
                is_ragged=False,
            )
        std, mean = torch.std_mean(fbank, dim=0)
        fbank = fbank.subtract(mean).divide(std)
        src = SequenceData(
            seqs=fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([fbank.shape[0]]),
            is_ragged=False,
        )
        
        text_generation_opts, unit_generation_opts = set_generation_opts(s_config)
        if tgtm == "seamless_expressivity":
            text_output, unit_output = translator.predict(
                                    src,
                                    "s2tt",
                                    use_target_lang[0],
                                    text_generation_opts=text_generation_opts,
                                    unit_generation_opts=unit_generation_opts,
                                    unit_generation_ngram_filtering=s_config.unit_generation_ngram_filtering,
                                    duration_factor=s_config.duration_factor,
                                    prosody_encoder_input=src_gcmvn,
                                )
        else:
            text_output, unit_output = translator.predict(
                                        src,
                                        "s2tt",
                                        use_target_lang[0],
                                        text_generation_opts=text_generation_opts,
                                        unit_generation_opts=unit_generation_opts,
                                        unit_generation_ngram_filtering=s_config.unit_generation_ngram_filtering,
                                        duration_factor=s_config.duration_factor,
                                    )
        untarget_token = use_token_encoder[0](str(text_output[0])).to(device)
        untargets = untarget_token.unsqueeze(0)

        if cycle_flag:
            if use_target_lang[0] != "eng" or len(use_target_lang) < 4:
                raise Exception( "--tgtl must contains eng, and the number of languages should not be fewer than 4")
            new_start_phrase = []
            for now_tl_index in range(1, len(use_target_lang)):
                start_phrase = phrase
                for i in range(2):
                    text_output_now, _ = text_translator.predict(
                        str(start_phrase),
                        "t2tt",
                        tgt_lang = use_target_lang[now_tl_index],
                        src_lang = s_config.tgt_lang,
                        text_generation_opts=text_generation_opts,
                        unit_generation_opts=unit_generation_opts,
                        unit_generation_ngram_filtering=s_config.unit_generation_ngram_filtering,
                    )
                    text_output_now = str(text_output_now[0])
                    logging.info("#"*20 + "translated:\t" + text_output_now)
                    text_output_two, _ = text_translator.predict(
                        text_output_now,
                        "t2tt",
                        tgt_lang = s_config.tgt_lang,
                        src_lang = use_target_lang[now_tl_index],
                        text_generation_opts=text_generation_opts,
                        unit_generation_opts=unit_generation_opts,
                        unit_generation_ngram_filtering=s_config.unit_generation_ngram_filtering,
                    )
                    start_phrase = str(text_output_two[0])
                    new_start_phrase.append(start_phrase)
                    logging.info("#"*20 + "new_start:\t" + start_phrase)
            new_phrase, count = find_most_frequent_string(new_start_phrase)
            logging.info("#"*20 + "new_start:\t" + new_phrase + "\tcount:" + str(count))
            phrase = new_phrase
        
        with open(os.path.join(outdir, "target.txt"), "w") as f:
            f.write(phrase)
            f.close()
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
        self.target_prefix_seqs = target_prefix_seq.expand(acoustic_seqs.shape[0], -1)
        min_prompt_len, min_prompt_idx = self.target_prefix_seqs.size(1), 0 # 2,0
        self.max_prompt_len, max_prompt_idx = self.target_prefix_seqs.size(1), 0
        max_gen_len = text_generation_opts.soft_max_seq_len
        a_term, b_term = max_gen_len
        max_source_len = acoustic_seqs.size(1)
        max_gen_len = int(a_term * max_source_len + b_term)
        max_seq_len = 1024
        self.max_seq_len = min(max_seq_len, self.max_prompt_len + max_gen_len)
        self.num_prompts = self.target_prefix_seqs.size(0)
        self.all_logits = torch.randn(target_phase_list[0].shape[1],1,256102)

        self.targets = torch.cat((torch.tensor([self.generator.s2t_converter.generator.model.target_vocab_info.eos_idx]).unsqueeze(0).to(device), target_phase_list[0]), dim=1)
        self.untargets = torch.cat((torch.tensor([self.generator.s2t_converter.generator.model.target_vocab_info.eos_idx]).unsqueeze(0).to(device), untargets), dim=1)
        self.targets_more_list = []
        for now_tl_index in range(1, len(use_target_lang)):
            self.targets_more_list.append(\
                torch.cat((torch.tensor([self.generator.s2t_converter.generator.model.target_vocab_info.eos_idx]).unsqueeze(0).to(device), target_phase_list[now_tl_index]), dim=1))

        self.min_prompt_len = min_prompt_len
        self.prefill_len = min_prompt_len # 
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
                acoustic_seqs, padding_mask
            )

        ce_loss = 0
        texts = ""
        if teacher_forcing:
            ######################################################################### teacher forcing
            texts_token = []
            temp_state_bag = IncrementalStateBag(self.max_seq_len)
            temp_seqs = self.targets
            for j in range(1, temp_seqs.shape[1]):
                decoder_output, decoder_padding_mask = self.generator.s2t_converter.generator.model.decode(
                        temp_seqs[:, j-1:j],
                        None,  # We never use PAD in incremental decoding.
                        encoder_output,
                        encoder_padding_mask,
                        state_bag=temp_state_bag,
                    )
                model_output = self.generator.s2t_converter.generator.model.project(decoder_output, decoder_padding_mask)
                temp_state_bag.increment_step_nr()
                logits = model_output.logits # torch.Size([1, 1, 256102])
                lprobs = log_softmax(logits, dim=-1, dtype=dtype)
                if j > 1:
                    if use_untarget:
                        if self.untargets[:, j] == temp_seqs[:, j]:
                            ce_loss = ce_loss + ce_loss_func(lprobs.squeeze(1), temp_seqs[:, j])
                        else:
                            ce_loss = ce_loss + ce_loss_func(lprobs.squeeze(1), temp_seqs[:, j]) - ce_loss_func(lprobs.squeeze(1), self.untargets[:, j]) * 0.01
                    else:
                        ce_loss = ce_loss + ce_loss_func(lprobs.squeeze(1), temp_seqs[:, j])
                top_scores, top_indices = torch.topk(lprobs, k=1)
                texts_token.append(top_indices)
                temp_state_bag.reorder((top_indices // lprobs.shape[-1]).squeeze(0).squeeze(0))
            texts = [self.generator.s2t_converter.text_decoder(torch.cat(texts_token, dim=1).squeeze().squeeze())]
            return_texts_token = torch.cat(texts_token, dim=1).squeeze().squeeze()

            ##### more lang
            for temp_seqs in self.targets_more_list:
                texts_token = []
                temp_state_bag = IncrementalStateBag(self.max_seq_len)
                for j in range(1, temp_seqs.shape[1]):
                    decoder_output, decoder_padding_mask = self.generator.s2t_converter.generator.model.decode(
                            temp_seqs[:, j-1:j],
                            None,  # We never use PAD in incremental decoding.
                            encoder_output,
                            encoder_padding_mask,
                            state_bag=temp_state_bag,
                        )
                    model_output = self.generator.s2t_converter.generator.model.project(decoder_output, decoder_padding_mask)
                    temp_state_bag.increment_step_nr()
                    logits = model_output.logits # torch.Size([1, 1, 256102])
                    lprobs = log_softmax(logits, dim=-1, dtype=dtype)
                    if j > 1:
                        ce_loss = ce_loss + ce_loss_func(lprobs.squeeze(1), temp_seqs[:, j])
                    top_scores, top_indices = torch.topk(lprobs, k=1)
                    texts_token.append(top_indices)
                    temp_state_bag.reorder((top_indices // lprobs.shape[-1]).squeeze(0).squeeze(0))
                texts.append(self.generator.s2t_converter.text_decoder(torch.cat(texts_token, dim=1).squeeze().squeeze()))
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




class Attack:
    def __init__(self, tgtm, cycle_flag, audio, phrase, freq_min, freq_max, batch_size, learning_rate, weight_decay, outdir, bp, psy):
        self.bp = bp
        self.psy = psy
        self.outdir = outdir

        assert len(audio.shape) == 1
        self.phrase = phrase 
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.batch_size = batch_size

        # Store arguments as constant tensors.
        self.original = torch.tensor(audio, requires_grad=False, device=device, dtype=torch.float32).unsqueeze(0)
        # psychoacoustic added
        # pdb.set_trace()
        theta, original_max_psd = masker._compute_masking_threshold(self.original.squeeze().detach().cpu().numpy())
        self.theta = torch.FloatTensor(theta.transpose(1, 0)).to(device)
        self.original_max_psd = torch.FloatTensor([original_max_psd]).to(device)
        
        self.delta = torch.tensor(np.random.normal(0, np.sqrt(np.abs(audio).mean()*0.1), audio.shape).astype(np.float32), requires_grad=True, device=device, dtype=torch.float32)

        # Create a band pass filter to be applied to the perturbation.
        freq = np.fft.rfftfreq(audio.shape[0], 1.0 / Fs)
        bp_filter = ((self.freq_min < freq) & (freq < self.freq_max)).astype(np.int32)
        self.bp_filter = torch.tensor(bp_filter.astype(np.complex64), requires_grad=False, device=device, dtype=dtype)
        self.optimizer = torch.optim.AdamW([self.delta], lr=learning_rate, weight_decay=weight_decay)
        self.ctc_loss_func = torch.nn.CTCLoss()
        self.noise_ratio = torch.tensor(np.ones((1, ), dtype=np.float32), requires_grad=True, device=device, dtype=dtype)
        self.get_logits = get_logits(tgtm, cycle_flag, self.outdir, self.original, self.phrase)
        self.text_phase_list = self.get_logits.text_phase_list

    def run_step(self, itr):
        # Apply the filter for the delta to simulate the real-world and create an adversarial example.
        if self.bp == 1:
            delta_filtered = torch.fft.irfft(torch.fft.rfft(eps * self.delta.tanh()) * self.bp_filter)
        else:
            delta_filtered = eps * self.delta.tanh()
        padding = self.original.size(-1) - delta_filtered.size(-1)
        delta_filtered_padded = F.pad(delta_filtered, (0, padding))
        ae_input = self.original + delta_filtered_padded

        # Change filters to apply dynamically
        # imp_indices = torch.tensor(np.zeros(imp_indices, dtype=np.int32), requires_grad=True)
        # apply_filters = torch.gather(self.imp_filters, 0, imp_indices)
        # # Convolve the impulse responses to the input
        # fft_length = torch.tensor(np.array([self.nfft], dtype=np.int32), requires_grad=False)
        # ae_frequency = torch.rfft(ae_input, fft_length=[self.nfft]) * apply_filters
        # ae_convolved = torch.irfft(ae_frequency, fft_length=[self.nfft])[:, :self.conv_length]
        ae_convolved = ae_input

        # Normalize the convolved audio
        max_audio = torch.max(torch.abs(ae_convolved), dim=1, keepdim=True).values
        ae_transformed = ae_convolved / max_audio * torch.max(torch.abs(ae_input))

        # Add a tiny bit of noise to help make sure that we can clip our values to 16-bit integers and not break things.
        small_noise = torch.randn(ae_transformed.size(), dtype=ae_transformed.dtype, device=ae_transformed.device)
        small_noise = small_noise * torch.tensor(1 - self.noise_ratio, dtype=ae_transformed.dtype, device=ae_transformed.device)
        final_input = torch.clamp((ae_transformed + small_noise)*(2 ** 15 - 1), min=-2 ** 15, max=2 ** 15 - 1) / (2 ** 15 - 1)

        logits, texts, ce_loss, return_texts_token = self.get_logits.logits(final_input, True)
        if (itr) % 100 == 0:
            soundfile.write(os.path.join(self.outdir, str(texts[0])+"_.wav"), final_input.squeeze().detach().cpu().numpy(), samplerate=Fs)

        if self.psy == 1:
            psy_loss = masker.batch_forward_2nd_stage(
                                                local_delta_rescale=delta_filtered_padded.unsqueeze(0),
                                                theta_batch=self.theta.unsqueeze(0),
                                                original_max_psd_batch=self.original_max_psd.unsqueeze(0),
                                                )
        else:
            psy_loss = 0.
        sum_loss = ce_loss + psy_loss*1e-7

        self.optimizer.zero_grad()
        sum_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.delta, max_norm=1.0)
        logging.info('Grad: %s' %(np.array_str(self.delta.grad.detach().cpu().numpy(), max_line_width=120)))
        self.optimizer.step()

        decoded = [str(te) for te in texts]
        return decoded, logits, ce_loss, ae_transformed, ae_input, delta_filtered, return_texts_token


    def attack(self, outdir, num_iterations=5000):
        prefix = ''.join([random.choice(string.ascii_lowercase) for _ in range(3)])
        time_last, time_start = time.time(), time.time()

        for itr in range(num_iterations + 1):
            # Actually do the optimization step
            decoded, logits, ce_loss, ae_transformed, ae_input, delta_filtered, return_texts_token = self.run_step(itr)

            # Report progress
            ce_loss = [ce_loss]
            logging.info('Iter: %d, Elapsed Time: %.3f, Iter Time: %.3f\n\tLosses: %s\n\tDelta: %s' % \
                  (itr, time.time() - time_start, time.time() - time_last, ' '.join('% 6.2f' % x for x in ce_loss), np.array_str(delta_filtered.detach().cpu().numpy(), max_line_width=120)))
            time_last = time.time()

            # logging.info out some debug information every 5 iterations.
            if itr % 5 == 0:
                logging.info('Recognition:\n\t' + '\n\t'.join(decoded))
                logging.info(f"{list(return_texts_token.detach().cpu().numpy())}")
            if set(decoded) == set(self.text_phase_list):
                # Get the current constant
                ratio = self.noise_ratio
                logging.info('=> It: %d, Noise Ratio: %.3f' % (itr, 1.0 - ratio[0]))
                # Update with the new noise
                self.noise_ratio = ratio * noise_step_ratio
                logging.info(set(decoded))
                if itr % 1000 == 0: 
                    itr2 = itr + 1
                else:
                    itr2 = itr
                wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr2)), Fs, np.array(np.clip(np.round(ae_input.squeeze(0).detach().cpu().numpy()*(2 ** 15 - 1)), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
                break

            if itr % 1000 == 0:
                wav.write(os.path.join(outdir, '%s-adv-%d.wav' % (prefix, itr)), Fs, np.array(np.clip(np.round(ae_input.squeeze(0).detach().cpu().numpy()*(2 ** 15 - 1)), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))


def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support generating one adversarial example at a time.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest='input', required=True,
                        help='Input audio .wav file, at {fs}Hz'.format(fs=Fs))
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
    parser.add_argument('--iterations', type=int, required=False, default=5000,
                        help='Maximum number of iterations of gradient descent')
    parser.add_argument('--eps', type=float, required=False, default=0.1,
                        help='noise strength')
    parser.add_argument('--bp', type=int, required=False, default=1,
                        help='if use bandpass')
    parser.add_argument('--psy', type=int, required=False, default=0,
                        help='if use psy')
    parser.add_argument('--noise', type=float, required=False, default=1.,
                        help='noise tep ratio, 1. means no noise')
    parser.add_argument('--uut', type=int, required=False, default=0,
                        help='if use untarget')
    parser.add_argument('--tgtl', type=str, required=False, default="eng,cmn,deu,fra",
                        help='target language')
    parser.add_argument('--tgtm', type=str, required=False, default="seamlessM4T_large",
                        help='target model')
    parser.add_argument('--gated_model_dir', type=str, required=False, default="facebook/seamless-expressive",
                        help='local model path')
    parser.add_argument('--cycle', type=int, required=False, default=0,
                        help='if use cycle opt')
    args = parser.parse_args()
    
    global eps
    eps = args.eps
    
    global noise_step_ratio
    noise_step_ratio = args.noise
    global use_untarget
    use_untarget = args.uut
    global use_target_lang
    use_target_lang = args.tgtl.split(",")
    global use_token_encoder
    use_token_encoder = []


    global s_config
    global gcmvn_mean
    global gcmvn_std
    global translator

    s_config = Seamless_configuration()
    s_config.tgt_lang = "eng"

    logging.info(f"Running inference on {device=} with {dtype=}.")
    
    global text_translator

    if args.tgtm == "seamless_expressivity":
        s_config.model_name = "seamless_expressivity"
        s_config.vocoder_name = "vocoder_pretssel"
        _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(s_config.vocoder_name)
        gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
        gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)
        s_config.gated_model_dir = Path(args.gated_model_dir)
        add_gated_assets(s_config.gated_model_dir)
        text_translator = Translator(
            "seamlessM4T_v2_large",
            vocoder_name_or_card=None,
            device=device,
            dtype=dtype,
        )
        translator = Translator(
            s_config.model_name,
            vocoder_name_or_card=None,
            device=device,
            dtype=dtype,
        )
    else:
        text_translator = Translator(
                args.tgtm,
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
    
    # Load the inputs that we're given
    a,fs = librosa.load(args.input, sr=Fs)
    audio = a
    assert fs == Fs

    # Set up the attack class and run it
    attack = Attack(args.tgtm, args.cycle, audio, args.target, freq_min=args.freq_min, freq_max=args.freq_max,
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
    attack.attack(outdir=args.out, num_iterations=args.iterations)
    

if __name__ == '__main__':
    main()