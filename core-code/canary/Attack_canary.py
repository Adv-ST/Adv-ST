import sys
path_to_add = './'
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import pdb
import os
os.environ['TERM'] = 'xterm'

from psy import Masker
import argparse

import torch
from pathlib import Path

from dataclasses import dataclass
from torch import Tensor
torch.autograd.set_detect_anomaly(True)
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


# Sampling rate of the input files
Fs = 16000
BEAM_SIZE = 1

def ctc_beam_search_decoder():
    return 0
masker = Masker(device = device)
ce_loss_func = torch.nn.CrossEntropyLoss()
my_ctc_loss_fn = torch.nn.CTCLoss()

from canary_utils import canary_predict, lang_name_3_to_2
class get_logits():
    def __init__(self, wav, phrase):
        original_text = canary_predict(canary_model, wav, source_lang, source_lang, "translate")
        logging.info(original_text)
        untarget_token = original_text[0].y_sequence.to(device)
        self.untargets = untarget_token.unsqueeze(0)
        # target list
        ref_audio, _ = librosa.load(os.path.join("../core-code/canary/reference", phrase+".wav"), sr=Fs)
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
            input_wave_len = torch.Tensor([int(input_wave_tensor.shape[-1])]).cuda()
            # print(f"input_wave_tensor.requires_grad : {input_wave_tensor.requires_grad}")
            processed_signal, processed_signal_length = canary_model.preprocessor(
                input_signal=input_wave_tensor.cuda(), length=input_wave_len
            )
            # print(f"processed_signal.requires_grad after preprocessing: {processed_signal.requires_grad}")
            # print(f"processed_signal.shape after preprocessing: {processed_signal.shape}")
            log_probs, encoded_len, enc_states, enc_mask = canary_model(
                # input_signal=input_wave_tensor.cuda(), input_signal_length=input_wave_len,
                processed_signal=processed_signal, processed_signal_length=processed_signal_length)
            # print(f"enc_states.requires_grad after encoder : {enc_states.requires_grad}")
            tokenizer = canary_model.tokenizer
            texts = []
            return_texts_token = []
            ce_loss = 0
            for now_index in range(len(use_target_lang)):
                tgt_lang = use_target_lang[now_index]
                # print(f"input_wave_tensor.requires_grad : {input_wave_tensor.requires_grad}")
                target_decoder_output_token_sequence = self.targets_more_list[now_index]
                if task is None:
                    if src_lang == tgt_lang:
                        task_token = tokenizer.transcribe_id
                    else:
                        task_token = tokenizer.translate_id
                else: # task is not None
                    if task == 'transcribe':
                        task_token = tokenizer.transcribe_id
                    elif task == 'translate':
                        task_token = tokenizer.translate_id
                    else:
                        raise ValueError("task should be either 'transcribe' or 'translate'.")
                # build the prompt
                # check if mistakenly input 3-letter language name
                src_lang = lang_name_3_to_2(src_lang)
                tgt_lang = lang_name_3_to_2(tgt_lang)
                prompt = torch.Tensor(
                    [
                        [
                            tokenizer.bos_id,
                            tokenizer.to_language_id(src_lang),
                            task_token,
                            tokenizer.to_language_id(tgt_lang),
                            tokenizer.pnc_id,  # we do not consider the punctuation variations.
                        ]
                    ]
                ).to(torch.long).cuda()
                
                encoder_hidden_states=enc_states
                encoder_input_mask=enc_mask
                decoder_input_ids=prompt
                beam_search = canary_model.decoding.decoding.beam_search

                # below we start decoding
                tgt, batch_size, max_generation_length = beam_search._prepare_for_search(decoder_input_ids, encoder_hidden_states)
                # tgt shape [1,5], same as decoder_input_ids

                # pad profile tracks sequences ending with <eos> token to replace
                # everything after <eos> with <pad> token
                decoder_parameter = next(beam_search.decoder.parameters())
                pad_profile = torch.zeros(batch_size, 1).long().to(decoder_parameter.device)

                log_probs_results = []
                # log_probs shape: [1, 1, 4128], likely [beamsize, batchsize, vocab_size]
                log_probs, decoder_mems_list = beam_search._one_step_forward(tgt, encoder_hidden_states, encoder_input_mask, None, 0)
                # print(f"log_probs.shape: {log_probs.shape}")
                prompt_len = tgt.shape[-1]  # default is 5

                # get the log probs
                # first we concatenate the tgt with the given target_decoder_output_token_sequence
                # if the dim of target_decoder_output_token_sequence is 1, we unsqueeze it to 2
                if len(target_decoder_output_token_sequence.shape) == 1:
                    target_decoder_output_token_sequence = target_decoder_output_token_sequence.unsqueeze(0)
                tgt = target_decoder_output_token_sequence
                # if the target_decoder_output_token_sequence is empty, we only get the first log prob
                # so, we decode at least for one time. Note that the previous decoding is used to setting up the decoder_mems_list.
                texts_token = []
                for i in range(prompt_len, target_decoder_output_token_sequence.shape[-1]):
                    
                    log_probs, decoder_mems_list = beam_search._one_step_forward(
                        tgt[:, i-1:i], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i
                    )
                    if use_untarget:
                        if self.untargets[:, i] == target_decoder_output_token_sequence[:, i]:
                            ce_loss = ce_loss + ce_loss_func(log_probs.squeeze(1), target_decoder_output_token_sequence[:, i])
                        else:
                            ce_loss = ce_loss + ce_loss_func(log_probs.squeeze(1), target_decoder_output_token_sequence[:, i]) - \
                                ce_loss_func(log_probs.squeeze(1), self.untargets[:, i]) * 0.01
                    else:
                        ce_loss = ce_loss + ce_loss_func(log_probs.squeeze(1), target_decoder_output_token_sequence[:, i])
                    texts_token.append(torch.argmax(log_probs[:, -1], dim=-1, keepdim=True))
                    # if i > prompt_len + 1:
                    #     logging.info(canary_model.decoding.decode_tokens_to_str(torch.cat(texts_token, dim=1).cpu().squeeze().numpy().tolist()))
                texts.append(\
                    canary_model.decoding.strip_special_tokens(\
                    canary_model.decoding.decode_tokens_to_str(torch.cat(texts_token, dim=1).cpu().squeeze().numpy().tolist())
                    ))
                return_texts_token.append(torch.cat(texts_token, dim=1).squeeze().detach().cpu().numpy().tolist())

                # restore the model states
                canary_model.preprocessor.featurizer.dither = original_dither_value
                canary_model.preprocessor.featurizer.pad_to = original_pad_to_value
                # return log_probs_results
        logging.info("\n" + "*" * 40)
        logging.info(texts)
        return None, texts, ce_loss, return_texts_token



class Attack:
    def __init__(self, audio, phrase, freq_min, freq_max, batch_size, learning_rate, weight_decay, outdir, bp, psy):
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
        self.get_logits = get_logits(self.original, self.phrase)
        self.text_phase_list = self.get_logits.text_phase_list

    def run_step(self, itr):
        # Apply the filter for the delta to simulate the real-world and create an adversarial example.
        if self.bp == 1:
            # pdb.set_trace()
            delta_filtered = torch.fft.irfft(torch.fft.rfft(eps * self.delta.tanh()) * self.bp_filter)
        else:
            delta_filtered = eps * self.delta.tanh()
        # 计算需要填充的长度
        padding = self.original.size(-1) - delta_filtered.size(-1)
        # 填充delta_filtered以使其与original的长度相同
        delta_filtered_padded = F.pad(delta_filtered, (0, padding))
        # 创建对抗性样本
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
            # imp_losses[indice] = ce_loss.detach().cpu().numpy()

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
    parser.add_argument('--tgtl', type=str, required=False, default="eng,fra,deu,spa",
                        help='target language')
    parser.add_argument('--src_lang', type=str, required=False, default="eng",
                        help='source language')
    args = parser.parse_args()
    
    global source_lang
    source_lang = args.src_lang
    global eps
    eps = args.eps
    global noise_step_ratio
    noise_step_ratio = args.noise
    global use_untarget
    use_untarget = args.uut
    global use_target_lang
    use_target_lang = args.tgtl.split(",")
    global canary_model
    from nemo.collections.asr.models import EncDecMultiTaskModel
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').to(device)
    canary_model.eval()
    canary_model.encoder.freeze()
    canary_model.transf_decoder.freeze()
    for param in canary_model.parameters():
        param.requires_grad = False

    logging.info(f"Running inference on {device=} with {dtype=}.")
    logging.info(f'Command line: {args}')
    
    # Load the inputs that we're given
    a,fs = librosa.load(args.input, sr=Fs)
    audio = a
    assert fs == Fs

    # Set up the attack class and run it
    attack = Attack(audio, args.target, freq_min=args.freq_min, freq_max=args.freq_max,
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
        
    with open(os.path.join(args.out, "target.txt"), "w") as f:
        f.write(args.target)
        f.close()
    attack.attack(outdir=args.out, num_iterations=args.iterations)
    

if __name__ == '__main__':
    main()