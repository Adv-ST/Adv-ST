import sys
path_to_add = '../'
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
import pdb
import os
os.environ['TERM'] = 'xterm'
BEAM_SIZE = 1

from psy import Masker
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from torch import Tensor

torch.autograd.set_detect_anomaly(True)
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float32
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
from typing import Sequence
from torch.nn.functional import log_softmax

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

from canary_utils import canary_predict, lang_name_3_to_2


class get_logits():
    def __init__(self, phrase):
        # target list
        ref_audio, _ = librosa.load(os.path.join("../../core-code/canary/reference", phrase+".wav"), sr=Fs)
        ref_audio = torch.from_numpy(ref_audio).unsqueeze(0).to(device)
        self.targets_more_list = []
        self.text_phase_list = []
        for now_tl_index in range(len(use_target_lang)):
            now_text = canary_predict(canary_model, ref_audio, source_lang, use_target_lang[now_tl_index], "translate")
            self.text_phase_list.append(now_text[0].text)
            self.targets_more_list.append(now_text[0].y_sequence.unsqueeze(0).to(device))

    def logits(self, wav, teacher_forcing):
        """
        Compute the logits for a given waveform.
        """
        task = "translate"
        src_lang = source_lang
        if teacher_forcing:
            input_wave_tensor = wav
            original_dither_value = canary_model.preprocessor.featurizer.dither
            original_pad_to_value = canary_model.preprocessor.featurizer.pad_to
            canary_model.preprocessor.featurizer.dither = 0.0
            canary_model.preprocessor.featurizer.pad_to = 0
            input_wave_len = torch.Tensor([int(input_wave_tensor.shape[-1])
                                           ]).cuda()
            # print(f"input_wave_tensor.requires_grad : {input_wave_tensor.requires_grad}")
            processed_signal, processed_signal_length = canary_model.preprocessor(
                input_signal=input_wave_tensor.cuda(), length=input_wave_len)
            # print(f"processed_signal.requires_grad after preprocessing: {processed_signal.requires_grad}")
            # print(f"processed_signal.shape after preprocessing: {processed_signal.shape}")
            log_probs, encoded_len, enc_states, enc_mask = canary_model(
                # input_signal=input_wave_tensor.cuda(), input_signal_length=input_wave_len,
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length)
            # print(f"enc_states.requires_grad after encoder : {enc_states.requires_grad}")
            tokenizer = canary_model.tokenizer
            texts = []
            return_texts_token = []
            ce_loss = 0
            for now_index in range(len(use_target_lang)):
                tgt_lang = use_target_lang[now_index]
                # print(f"input_wave_tensor.requires_grad : {input_wave_tensor.requires_grad}")
                target_decoder_output_token_sequence = self.targets_more_list[
                    now_index]
                if task is None:
                    if src_lang == tgt_lang:
                        task_token = tokenizer.transcribe_id
                    else:
                        task_token = tokenizer.translate_id
                else:  # task is not None
                    if task == 'transcribe':
                        task_token = tokenizer.transcribe_id
                    elif task == 'translate':
                        task_token = tokenizer.translate_id
                    else:
                        raise ValueError(
                            "task should be either 'transcribe' or 'translate'."
                        )
                # build the prompt
                # check if mistakenly input 3-letter language name
                src_lang = lang_name_3_to_2(src_lang)
                tgt_lang = lang_name_3_to_2(tgt_lang)
                prompt = torch.Tensor([[
                    tokenizer.bos_id,
                    tokenizer.to_language_id(src_lang),
                    task_token,
                    tokenizer.to_language_id(tgt_lang),
                    tokenizer.
                    pnc_id,  # we do not consider the punctuation variations.
                ]]).to(torch.long).cuda()

                encoder_hidden_states = enc_states
                encoder_input_mask = enc_mask
                decoder_input_ids = prompt
                beam_search = canary_model.decoding.decoding.beam_search

                # below we start decoding
                tgt, batch_size, max_generation_length = beam_search._prepare_for_search(
                    decoder_input_ids, encoder_hidden_states)
                # tgt shape [1,5], same as decoder_input_ids

                # pad profile tracks sequences ending with <eos> token to replace
                # everything after <eos> with <pad> token
                decoder_parameter = next(beam_search.decoder.parameters())
                pad_profile = torch.zeros(batch_size, 1).long().to(
                    decoder_parameter.device)

                log_probs_results = []
                # log_probs shape: [1, 1, 4128], likely [beamsize, batchsize, vocab_size]
                log_probs, decoder_mems_list = beam_search._one_step_forward(
                    tgt, encoder_hidden_states, encoder_input_mask, None, 0)
                # print(f"log_probs.shape: {log_probs.shape}")
                prompt_len = tgt.shape[-1]  # default is 5

                # get the log probs
                # first we concatenate the tgt with the given target_decoder_output_token_sequence
                # if the dim of target_decoder_output_token_sequence is 1, we unsqueeze it to 2
                if len(target_decoder_output_token_sequence.shape) == 1:
                    target_decoder_output_token_sequence = target_decoder_output_token_sequence.unsqueeze(
                        0)
                tgt = target_decoder_output_token_sequence
                # if the target_decoder_output_token_sequence is empty, we only get the first log prob
                # so, we decode at least for one time. Note that the previous decoding is used to setting up the decoder_mems_list.
                texts_token = []
                for i in range(prompt_len,
                               target_decoder_output_token_sequence.shape[-1]):

                    log_probs, decoder_mems_list = beam_search._one_step_forward(
                        tgt[:, i - 1:i], encoder_hidden_states,
                        encoder_input_mask, decoder_mems_list, i)
                    
                    ce_loss = ce_loss + ce_loss_func(
                        log_probs.squeeze(1),
                        target_decoder_output_token_sequence[:, i])
                    texts_token.append(
                        torch.argmax(log_probs[:, -1], dim=-1, keepdim=True))
                texts.append(\
                    canary_model.decoding.strip_special_tokens(\
                    canary_model.decoding.decode_tokens_to_str(torch.cat(texts_token, dim=1).cpu().squeeze().numpy().tolist())
                    ))
                return_texts_token.append(
                    torch.cat(texts_token,
                              dim=1).squeeze().detach().cpu().numpy().tolist())
                # restore the model states
                canary_model.preprocessor.featurizer.dither = original_dither_value
                canary_model.preprocessor.featurizer.pad_to = original_pad_to_value
                # return log_probs_results
        logging.info("\n" + "*" * 40)
        logging.info(texts)
        return None, texts, ce_loss, return_texts_token


def kl_divergence(latents):
    mean, logvar = torch.mean(latents), torch.log(
        torch.var(latents, unbiased=False) + 1e-8)
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
    def __init__(self, impulse, phrase, freq_min, freq_max, batch_size,
                 learning_rate, weight_decay, outdir, bp, psy):
        audio = torch.randn(Fs*UN_LEN).numpy()*0.005
        self.bp = bp
        self.psy = psy
        self.outdir = outdir
        assert len(audio.shape) == 1
        self.phrase = phrase
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.impulse_size = 1
        self.batch_size = batch_size

        self.impulse = torch.tensor(impulse, dtype=torch.float32).to(device)
        
        self.tango = tango_model
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
        self.latents = self.tango.model.prepare_latents(
            1, inference_scheduler, num_channels_latents, torch.float32,
            device).requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
                    [self.latents] + \
                    [self.tango.model.PE.global_time_embedding.angles] +\
                    [self.tango.model.FME.translation_bias] + [self.tango.model.FME.angles],
                    lr=learning_rate,
                    weight_decay=weight_decay,
        )

        self.ctc_loss_func = torch.nn.CTCLoss()

        self.noise_ratio = torch.tensor(np.ones((1, ), dtype=np.float32),
                                        requires_grad=True,
                                        device=device,
                                        dtype=dtype)

        self.get_logits = get_logits(self.phrase)
        self.text_phase_list = self.get_logits.text_phase_list
        with torch.no_grad():
            self.beats, self.chords, self.chords_times = self.tango.music_model.generate(
                self.initial_prompt)
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
        latents = self.tango.model.inference2(
            self.latents, [self.initial_prompt], self.beats, [self.chords],
            [self.chords_times], self.tango.scheduler, self.num_steps,
            3.0)[:, :, :64, :]

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
        
        logits, texts, ce_loss, return_texts_token = self.get_logits.logits(
            final_input, True)
        if (itr) % 10 == 0:
            soundfile.write(os.path.join(self.outdir,
                                         str(texts[0]) + "_.wav"),
                            final_input.squeeze().detach().cpu().numpy(),
                            samplerate=Fs)

        psy_loss = kl_divergence(self.latents)
        
        sum_loss = ce_loss + psy_loss

        self.optimizer.zero_grad()
        sum_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latents, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(list(
            self.tango.model.text_encoder.get_input_embeddings().parameters()),
                                       max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            self.tango.model.PE.global_time_embedding.angles, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.tango.model.FME.translation_bias,
                                       max_norm=1.0)

        logging.info('Grad 1: %s' %(np.array_str(self.latents.grad[0][0][0].detach().cpu().numpy(), max_line_width=120)))
        logging.info('Grad 2: %s' %(np.array_str(self.tango.model.PE.global_time_embedding.angles.grad.detach().cpu().numpy(), max_line_width=120)))
        logging.info('Grad 3: %s' %(np.array_str(self.tango.model.FME.translation_bias.grad.detach().cpu().numpy(), max_line_width=120)))
        
        self.optimizer.step()

        decoded = [str(te) for te in texts]
        return decoded, logits, ce_loss, final_input, final_input, wave[0], return_texts_token


    def attack(self, outdir, num_iterations=5):
        # Create misc variables
        prefix = ''.join(
            [random.choice(string.ascii_lowercase) for _ in range(3)])
        time_last, time_start = time.time(), time.time()

        # We'll make a bunch of iterations of gradient descent here
        itr = 0
        most_flag = False
        for itr_ in range(num_iterations + 1):
            if most_flag: break
            for audio in train_loader:
                indice = np.random.choice(self.impulse.shape[0])
                audio = audio["matrix"]
                itr += 1
                if itr > 12000:
                    most_flag = True
                    break
                # Actually do the optimization step
                decoded, logits, ctcloss, ae_transformed, ae_input, delta_filtered, return_texts_token = self.run_step(
                    indice, itr, audio)

                # Report progress
                ctcloss = [ctcloss]
                logging.info('Iter: %d, Elapsed Time: %.3f, Iter Time: %.3f\n\tLosses: %s\n\tDelta: %s' % \
                    (itr, time.time() - time_start, time.time() - time_last, ' '.join('% 6.2f' % x for x in ctcloss), np.array_str(delta_filtered.detach().cpu().numpy(), max_line_width=120)))
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
    parser.add_argument('--in',
                        type=str,
                        dest='input',
                        required=False,
                        help='Input audio .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--imp',
                        type=str,
                        dest='impulse',
                        nargs='+',
                        required=True,
                        help='Input impulse response .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--target',
                        type=str,
                        required=True,
                        help='Target semantic')
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help='Directory for saving intermediate files')
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size for generation')
    parser.add_argument('--freq_min',
                        type=int,
                        required=False,
                        default=1000,
                        help='Lower limit of band pass filter for adversarial noise')
    parser.add_argument('--freq_max',
                        type=int,
                        required=False,
                        default=4000,
                        help='Higher limit of band pass filter for adversarial noise')
    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=0.1,
                        help='Learning rate for optimization')
    parser.add_argument('--decay',
                        type=float,
                        required=False,
                        default=0.001,
                        help='Weight decay for optimization')
    parser.add_argument('--iterations',
                        type=int,
                        required=False,
                        default=5,
                        help='Maximum number of iterations of gradient descent')
    parser.add_argument('--bp',
                        type=int,
                        required=False,
                        default=0,
                        help='if use bandpass')
    parser.add_argument('--psy',
                        type=int,
                        required=False,
                        default=0,
                        help='if use psy')
    parser.add_argument('--noise',
                        type=float,
                        required=False,
                        default=1.,
                        help='noise tep ratio')
    parser.add_argument('--tgtl',
                        type=str,
                        required=False,
                        default="eng,fra,deu,spa",
                        help='target language')
    parser.add_argument('--src_lang',
                        type=str,
                        required=False,
                        default="eng",
                        help='source language')
    parser.add_argument('--speech_pth', type=str, required=True, default="../../core-code/LibriSpeech_wav/",
                        help='speech dataset path')
    args = parser.parse_args()

    global source_lang
    source_lang = args.src_lang
    global noise_step_ratio
    noise_step_ratio = args.noise
    global use_target_lang
    use_target_lang = args.tgtl.split(",")
    global canary_model
    
    from nemo.collections.asr.models import EncDecMultiTaskModel
    load = True
    while load:
        try:
            canary_model = EncDecMultiTaskModel.from_pretrained(
                'nvidia/canary-1b').to(device)
            load = False
        except Exception as e:
            logging.info(e)
            continue
    canary_model.eval()
    canary_model.encoder.freeze()
    canary_model.transf_decoder.freeze()
    for param in canary_model.parameters():
        param.requires_grad = False
        
        
    global train_loader
    train_set = my_dataset(args.speech_pth, flag="train", leng=Fs*2)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    global tango_model
    from mustango import Mustango
    tango_model = Mustango("declare-lab/mustango", device)

    logging.info(f"Running inference on {device=} with {dtype=}.")
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
    attack = Attack(irs,
                    args.target,
                    freq_min=args.freq_min,
                    freq_max=args.freq_max,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    weight_decay=args.decay,
                    outdir=args.out,
                    bp=args.bp,
                    psy=args.psy)
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
