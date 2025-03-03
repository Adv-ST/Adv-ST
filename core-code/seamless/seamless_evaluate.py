import os
import argparse
import librosa
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from fairseq2.data import Collater
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)
from seamless_communication.inference import SequenceGeneratorOptions
from fairseq2.generation import NGramRepeatBlockProcessor
from fairseq2.memory import MemoryBlock
from huggingface_hub import snapshot_download
from seamless_communication.inference import Translator, SequenceGeneratorOptions
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.cli.expressivity.predict.pretssel_generator import PretsselGenerator
import torchaudio
import numpy as np
from pypesq import pesq
from resemblyzer import preprocess_wav, VoiceEncoder
from fairseq2.data import SequenceData
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from sentence_transformers import SentenceTransformer, util
from seamless_communication.store import add_gated_assets
from pathlib import Path
from typing import Tuple
import re
import warnings
warnings.filterwarnings("ignore")

BEAM_SIZE = 1
AUDIO_SAMPLE_RATE = 16000
MAX_INPUT_AUDIO_LENGTH = 10  # in seconds
bert_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli', trust_repo='check')
roberta.eval()  # disable dropout for evaluation


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32
    
resamber_encoder = VoiceEncoder(device=device)

m4t_translator = Translator(
        model_name_or_card="seamlessM4T_v2_large",
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )

convert_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )


def my_fbank_extracter(waveform, sample_rate, num_mel_bins=80, waveform_scale=2**15, 
                      channel_last=True, standardize=False, device='cpu', dtype=dtype):
    waveform = waveform * waveform_scale

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


def process_input(input_audio_path=None, gcmvn_flag=False):
    if isinstance(input_audio_path, str):
        wav, fs = librosa.load(input_audio_path, sr=16000)
        wav = torch.tensor(wav, requires_grad=False, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        wav = input_audio_path
    wav = wav.transpose(0, 1)
    my_fbank = my_fbank_extracter(wav, 16000).squeeze(0)
    fbank = my_fbank.to(dtype)
    if gcmvn_flag:
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
    if gcmvn_flag:
        return src, src_gcmvn
    else:
        return src
    

def remove_trailing_punctuation(text):
   punctuations = [
       ".", "!", "?", ";", ":", ")", "]", "}",  
       "、", "。", "？", "！", "；", "：", "—",  
       "…", "《", "》", "（", "）", "【", "】",  
       "——", "·", "′", "〃", "″", "‵", "々",  
       "‖", "︵", "︶", "︿", "﹀", "﹂", "﹁", "﹃", "﹄",  
       "※", "×", "÷",  
       "→", "←", "↑", "↓",  
       "××", "√",  
       "…",  
       "‰", "′", "〃", "℃", "℅",  
       "℃", "℉",  
       "√", "∵", "∴",  
       "～", "≈",  
       "ⅰ", "ⅱ", "ⅲ", "ⅳ",  
       "①", "②", "③", "④",  
       "（注释）", "［注释］",  
       "『", "』",  
       "〈", "〉",  
       "《", "》",  
       "'", "'", """, """,  
       "′", "″", "‴",  
       "※", "§", "@", "&", "#", "☆", "★", "○", "◎", "◇", "△", "☆", "★", "※", "→", "←", "↑", "↓",  
   ]
   return text[:-1] if text and text[-1] in punctuations else text

def cosine_similarity(x, y):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def calculate_snr_pesq_vsim(original_path, modified_path):
    # original_waveform, rate_original = torchaudio.load(original_path)
    original_waveform, rate_original = librosa.load(original_path, sr=16000)
    original_waveform = torch.tensor(original_waveform).unsqueeze(0)
    modified_waveform, rate_modified = torchaudio.load(modified_path)
    if rate_original != rate_modified:
        raise ValueError("sample rate are not same")
    if original_waveform.shape[1] != modified_waveform.shape[1]:
        raise Exception(f"shapes are {original_waveform.shape} and {modified_waveform.shape}")
    original_data = original_waveform.numpy().flatten()
    modified_data = modified_waveform.numpy().flatten()
    noise = modified_data - original_data
    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    pesq_score = pesq(original_data, modified_data, rate_original)
    ## resamplyer similarity
    wav_a = preprocess_wav(original_waveform.cpu().squeeze(0).detach().numpy(), source_sr=rate_original)
    wav_b = preprocess_wav(modified_waveform.cpu().squeeze(0).detach().numpy(), source_sr=rate_original)
    embeds_a = resamber_encoder.embed_utterance(wav_a)
    embeds_b = resamber_encoder.embed_utterance(wav_b)
    utterance_sim_matrix = cosine_similarity(embeds_a, embeds_b)
    ## seamless expressive similarity
    gcmvn_fbank = convert_to_fbank({"waveform": original_waveform.transpose(0,1), "sample_rate": 16000,})["fbank"].subtract(gcmvn_mean).divide(gcmvn_std).unsqueeze(0)
    src_gcmvn = SequenceData(
        seqs=gcmvn_fbank,
        seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
        is_ragged=False,
    )
    prosody_input_seqs, prosody_padding_mask = get_seqs_and_padding_mask(src_gcmvn)
    prosody_encoder_out = pretssel_generator.pretssel_model.encoder_frontend.prosody_encoder(
        prosody_input_seqs,
        prosody_padding_mask,
    ).unsqueeze(1)
    gcmvn_fbank_adv = convert_to_fbank({"waveform": modified_waveform.transpose(0,1), "sample_rate": 16000,})["fbank"].subtract(gcmvn_mean).divide(gcmvn_std).unsqueeze(0)
    adv_gcmvn = SequenceData(
        seqs=gcmvn_fbank_adv,
        seq_lens=torch.LongTensor([gcmvn_fbank_adv.shape[0]]),
        is_ragged=False,
    )
    prosody_input_seqs_adv, prosody_padding_mask_adv = get_seqs_and_padding_mask(adv_gcmvn)
    prosody_encoder_out_adv = pretssel_generator.pretssel_model.encoder_frontend.prosody_encoder(
        prosody_input_seqs_adv,
        prosody_padding_mask_adv,
    ).unsqueeze(1)
    seamless_sim = cosine_similarity(prosody_encoder_out.detach().cpu().numpy(), prosody_encoder_out_adv.detach().cpu().numpy())
    
    return pesq_score, utterance_sim_matrix, seamless_sim, rate_original


def calculate_sim(original_path, modified_path):
    pesq_score, utterance_sim_matrix, seamless_sim, rate_original = calculate_snr_pesq_vsim(original_path, modified_path)
    return pesq_score, utterance_sim_matrix, seamless_sim


def calculate_text_sim(tgt_text, adv_text, tgt_text_eng, adv_text_eng):
    bert_embedding_tgt = bert_model.encode(tgt_text)
    bert_embedding_adv = bert_model.encode(adv_text)
    ESIM_tgt = util.pytorch_cos_sim(bert_embedding_tgt, bert_embedding_adv).item()
    NSCORE_tgt = torch.exp(roberta.predict('mnli', roberta.encode(tgt_text_eng, adv_text_eng))[0][2]).item()
    return ESIM_tgt, NSCORE_tgt


def get_adv_in_dir(sentence_path):
    max_iterations = 5000
    max_b, max_b_file = None, None
    pattern = re.compile(r"([a-zA-Z]{3})-adv-(\d+)\.wav$")
    iter_list = []
    for filename in os.listdir(sentence_path):
        match = pattern.match(filename)
        if match:
            b = int(match.group(2))
            iter_list.append(filename)
    iter_list = sorted(iter_list, key=lambda name: int(name.split('-')[2].split('.')[0]))
    for name in iter_list:
        if int(name.split('-')[2].split('.')[0])%1000 != 0: 
            return name
        elif int(name.split('-')[2].split('.')[0]) == max_iterations:
            return name
    if int(iter_list[-1].split('-')[2].split('.')[0]) >= 1000:
        logging.info(f"return {iter_list[-1]}")
        return iter_list[-1]
    raise Exception(f"no expected adv files in dir {sentence_path}")

def get_subdirectories(directory):
    try:
        items = os.listdir(directory)
        subdirectories = [os.path.join(directory, item) for item in items if os.path.isdir(os.path.join(directory, item))]
        return subdirectories
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def find_attack_folders(directory):
    attack_folders = []
    for dir_name in os.listdir(directory):
        dir_path = os.path.join(directory, dir_name)
        if os.path.isdir(dir_path) and dir_name.startswith("Attack"):
            attack_folders.append(dir_path)
    return attack_folders


def preprocess_audio(input_audio_path=None, output_audio_path=None, AUDIO_SAMPLE_RATE=16000, MAX_INPUT_AUDIO_LENGTH=10) -> None:
    arr, org_sr = torchaudio.load(input_audio_path)
    if org_sr != AUDIO_SAMPLE_RATE:
        raise Exception(f"sample rate is {org_sr}")
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if arr.shape[1] > max_length:
        raise Exception(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    if output_audio_path:
        torchaudio.save(output_audio_path, arr, sample_rate=AUDIO_SAMPLE_RATE)



@torch.no_grad()
def m4t(tgtm, mode, input_audio_path, source_language_code, eng_target_text, target_language_code, original_audio_path):
    if tgtm == "seamlessM4T_v2_large":
        vocoder_name = "vocoder_v2"
    else:
        vocoder_name = "vocoder_36langs"
    translator = Translator(
        model_name_or_card=tgtm,
        vocoder_name_or_card=vocoder_name,
        device=device,
        dtype=dtype,
    )
    text_generation_opts = SequenceGeneratorOptions(
            beam_size=BEAM_SIZE, soft_max_seq_len=(1, 200)
        )
    target_text_generation_opts = SequenceGeneratorOptions(
        beam_size=1,
        unk_penalty=torch.inf,
        soft_max_seq_len=(1, 200),
        step_processor=NGramRepeatBlockProcessor(
            ngram_size=10,
        ),
    )
    if mode == "perturbation" and original_audio_path is not None:
        source_sentences_eng, _ = translator.predict(
            input=process_input(original_audio_path),
            task_str="S2TT",
            src_lang=source_language_code,
            tgt_lang="eng",
            text_generation_opts = text_generation_opts
        )
        original_text_eng = str(source_sentences_eng[0])
        PESQ, vsim, vsim_expressive = calculate_sim(original_audio_path, input_audio_path)
    ### adv speech
    preprocess_audio(input_audio_path, None, AUDIO_SAMPLE_RATE, MAX_INPUT_AUDIO_LENGTH)
    text_output, speech_output = translator.predict(
        input=process_input(input_audio_path),
        task_str="S2ST",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
        text_generation_opts=text_generation_opts,
    )
    input_name, input_ext = os.path.splitext(os.path.basename(input_audio_path))
    input_name = target_language_code + "_trans_" + input_name
    output_filename = f"{input_name}{input_ext}"
    torchaudio.save(
        os.path.join(os.path.dirname(input_audio_path), output_filename),
        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
        sample_rate=speech_output.sample_rate,
    )
    text_out = str(text_output[0])
    logging.info(f"translate to target_language_code, translated audio saved at {os.path.join(os.path.dirname(input_audio_path), output_filename)}")
        
    if target_language_code == "eng":
        adv_text_eng = text_out
        target_text_tgt = target_text_eng = eng_target_text
    else:
        adv_text_eng, _ = m4t_translator.predict(
            text_out,
            "t2tt",
            "eng",
            src_lang = target_language_code,
            # text_generation_opts=m4t_text_generation_opts,
        )
        adv_text_eng = str(adv_text_eng[0])
        target_text_tgt, _ = translator.predict( # used when generating
            eng_target_text,
            "t2tt",
            tgt_lang = target_language_code,
            src_lang = "eng",
            text_generation_opts=target_text_generation_opts,
        )
        target_text_tgt = str(target_text_tgt[0])
        target_text_eng, _ = m4t_translator.predict(
            target_text_tgt,
            "t2tt",
            "eng",
            src_lang = target_language_code,
            # text_generation_opts=target_text_generation_opts,
        )
        target_text_eng = str(target_text_eng[0])
                        
    ESIM_tgt, NSCORE_tgt = calculate_text_sim(target_text_tgt, text_out, target_text_eng, adv_text_eng)
    
    if mode == "perturbation" and original_audio_path is not None:
        logging.info("\n" + "=" * 70)
        logging.info("AUDIO QUALITY METRICS")
        logging.info("-" * 70)
        logging.info(f"PESQ:              {PESQ:.4f}")
        logging.info(f"VSIM:              {vsim:.4f}")
        logging.info(f"VSIM-E:            {vsim_expressive:.4f}")
        logging.info("-" * 70)
        logging.info(f"Original Speech (ENG):    \"{original_text_eng}\"")
        logging.info("=" * 70)

    logging.info("\n" + "=" * 70)
    logging.info("TRANSLATION METRICS")
    logging.info("-" * 70)
    logging.info(f"Target Language:          {target_language_code}")
    logging.info(f"ESIM:                     {ESIM_tgt:.4f}")
    logging.info(f"NSCORE:                   {NSCORE_tgt:.4f}")
    logging.info("-" * 70)
    logging.info(f"Target Semantic (ENG):    \"{target_text_eng}\"")
    logging.info(f"Target Text ({target_language_code}):        \"{target_text_tgt}\"")
    logging.info(f"Adversarial Output:       \"{text_out}\"")
    logging.info(f"Adversarial Output (ENG): \"{adv_text_eng}\"")
    logging.info("=" * 70)
        
        
        
@torch.no_grad()
def expressive(tgtm, mode, input_audio_path, source_language_code, eng_target_text, target_language_code, original_audio_path):
    translator = Translator(
        model_name_or_card=tgtm,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
        apply_mintox=False,
    )
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=BEAM_SIZE,
        unk_penalty=torch.inf,
        soft_max_seq_len=(0, 200),
        step_processor=NGramRepeatBlockProcessor(
            ngram_size=10,
        ),
    )
    m4t_text_generation_opts = SequenceGeneratorOptions(
        beam_size=BEAM_SIZE,
        unk_penalty=torch.inf,
        soft_max_seq_len=(1, 200),
        step_processor=NGramRepeatBlockProcessor(
            ngram_size=10,
        ),
    )
    target_text_generation_opts = SequenceGeneratorOptions(
        beam_size=1,
        unk_penalty=torch.inf,
        soft_max_seq_len=(1, 200),
        step_processor=NGramRepeatBlockProcessor(
            ngram_size=10,
        ),
    )
    def remove_prosody_tokens_from_text(text):
        # filter out prosody tokens, there is only emphasis '*', and pause '='
        text = text.replace("*", "").replace("=", "")
        text = " ".join(text.split())
        return text
    
    if mode == "perturbation" and original_audio_path is not None:
        example_original = {}
        example_original["fbank"], example_original["gcmvn_fbank"] = process_input(original_audio_path, gcmvn_flag=True)
        source_sentences_eng, _ = m4t_translator.predict(
                                input=example_original["fbank"],
                                task_str="s2tt",
                                tgt_lang="eng",
                                text_generation_opts=m4t_text_generation_opts,
                            )
        original_text_eng = str(source_sentences_eng[0])
        PESQ, vsim, vsim_expressive = calculate_sim(original_audio_path, input_audio_path)
    ### adv speech
    example = {}
    example["fbank"], example["gcmvn_fbank"] = process_input(input_audio_path, gcmvn_flag=True)
    
    # get transcription for mintox
    source_sentences, _ = m4t_translator.predict(
        input=example["fbank"],
        task_str="S2TT",  # get source text
        tgt_lang=source_language_code,
        text_generation_opts=m4t_text_generation_opts,
    )
    source_text = str(source_sentences[0])
    prosody_encoder_input = example["gcmvn_fbank"]
    
    text_output, unit_output = translator.predict(
                        example["fbank"],
                        "S2ST",
                        tgt_lang=target_language_code,
                        src_lang=source_language_code,
                        text_generation_opts=text_generation_opts,
                        unit_generation_ngram_filtering=False,
                        duration_factor=1.0,
                        prosody_encoder_input=prosody_encoder_input,
                        src_text=source_text,  # for mintox check
                    )
    speech_output = pretssel_generator.predict(
                        unit_output.units,
                        tgt_lang=target_language_code,
                        prosody_encoder_input=prosody_encoder_input,
                    )
    input_name, input_ext = os.path.splitext(os.path.basename(input_audio_path))
    input_name = target_language_code + "_trans_" + input_name
    output_filename = f"{input_name}{input_ext}"
    torchaudio.save(
        os.path.join(os.path.dirname(input_audio_path), output_filename),
        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
        sample_rate=speech_output.sample_rate,
    )
    text_out = remove_prosody_tokens_from_text(str(text_output[0]))
    logging.info(f"translate to target_language_code, translated audio saved at {os.path.join(os.path.dirname(input_audio_path), output_filename)}")
        
    if target_language_code == "eng":
        adv_text_eng = text_out
        target_text_tgt = target_text_eng = eng_target_text
    else:
        adv_text_eng, _ = m4t_translator.predict(
            text_out,
            "t2tt",
            "eng",
            src_lang = target_language_code,
            # text_generation_opts=m4t_text_generation_opts,
        )
        adv_text_eng = remove_prosody_tokens_from_text(str(adv_text_eng[0]))
        target_text_tgt, _ = m4t_translator.predict( # used when generating
            eng_target_text,
            "t2tt",
            tgt_lang = target_language_code,
            src_lang = "eng",
            text_generation_opts=target_text_generation_opts,
        )
        target_text_tgt = remove_prosody_tokens_from_text(str(target_text_tgt[0]))
        target_text_eng, _ = m4t_translator.predict(
            target_text_tgt,
            "t2tt",
            "eng",
            src_lang = target_language_code,
            # text_generation_opts=target_text_generation_opts,
        )
        target_text_eng = remove_prosody_tokens_from_text(str(target_text_eng[0]))
                        
    ESIM_tgt, NSCORE_tgt = calculate_text_sim(target_text_tgt, text_out, target_text_eng, adv_text_eng)
    
    
    if mode == "perturbation" and original_audio_path is not None:
        logging.info("\n" + "=" * 70)
        logging.info("AUDIO QUALITY METRICS")
        logging.info("-" * 70)
        logging.info(f"PESQ:              {PESQ:.4f}")
        logging.info(f"VSIM:              {vsim:.4f}")
        logging.info(f"VSIM-E:            {vsim_expressive:.4f}")
        logging.info("-" * 70)
        logging.info(f"Original Speech (ENG):    \"{original_text_eng}\"")
        logging.info("=" * 70)

    logging.info("\n" + "=" * 70)
    logging.info("TRANSLATION METRICS")
    logging.info("-" * 70)
    logging.info(f"Target Language:          {target_language_code}")
    logging.info(f"ESIM:                     {ESIM_tgt:.4f}")
    logging.info(f"NSCORE:                   {NSCORE_tgt:.4f}")
    logging.info("-" * 70)
    logging.info(f"Target Semantic (ENG):    \"{target_text_eng}\"")
    logging.info(f"Target Text ({target_language_code}):        \"{target_text_tgt}\"")
    logging.info(f"Adversarial Output:       \"{text_out}\"")
    logging.info(f"Adversarial Output (ENG): \"{adv_text_eng}\"")
    logging.info("=" * 70)



def evaluate(tgtm, mode, input, speaker_lang, target, target_lang, original_audio):
    if tgtm in ["seamlessM4T_medium", "seamlessM4T_large", "seamlessM4T_v2_large"]:
        m4t(tgtm, mode, input, speaker_lang, target, target_lang, original_audio)
    elif tgtm in ["seamless_expressivity"]:
        expressive(tgtm, mode, input, speaker_lang, target, target_lang, original_audio)




if __name__ == "__main__":
    Fs = 16000
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--tgtm', type=str, required=False, 
                   choices=['seamlessM4T_medium', 'seamlessM4T_large', 'seamless_expressivity', 'seamlessM4T_v2_large'],
                   default="seamlessM4T_large",
                   help='Target model to use')
    parser.add_argument('--mode', type=str, required=True, choices=['perturbation', 'music'], \
                    help='Mode of attack: perturbation or music')
    parser.add_argument('--in', type=str, dest='input', required=True,
                    help='Input adversarial audio .wav file, at {fs}Hz'.format(fs=Fs))
    parser.add_argument('--speaker_lang', type=str, required=False, default="eng",
                    help='The language of the input audio, such as eng')
    parser.add_argument('--target_lang', type=str, required=False, default="eng",
                    help='The language of the translated audio, such as eng')
    parser.add_argument('--target_semantic', type=str, required=True,
                    help='Target semantic (in english)')
    parser.add_argument('--original_audio', type=str, required=False,
                    help='Path to the original audio .wav file at {fs}Hz. \
                            should be provided in perturbation mode.'.format(fs=Fs))
    parser.add_argument('--gated_model_dir', type=str, required=True, default="../facebook/seamless-expressive",
                        help='local model path')
    args = parser.parse_args()
    
    
    global pretssel_generator
    global gcmvn_mean
    global gcmvn_std
    MODEL_NAME = "seamless_expressivity"
    VOCODER_NAME = "vocoder_pretssel"
    unit_tokenizer = load_unity_unit_tokenizer(MODEL_NAME)
    add_gated_assets(Path(args.gated_model_dir))
    pretssel_generator = PretsselGenerator(
            VOCODER_NAME,
            vocab_info=unit_tokenizer.vocab_info,
            device=device,
            dtype=dtype,
        )
    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(VOCODER_NAME)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)
    
    
    if args.mode == 'perturbation' and args.original_audio is None:
        logging.info("No original audio provided.")    
    
    evaluate(args.tgtm, args.mode, args.input, args.speaker_lang, args.target_semantic, args.target_lang, args.original_audio)