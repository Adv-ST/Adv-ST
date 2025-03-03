# Code by whl, 2024/06/17
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import IPython.display as ipd
from torch.autograd import Variable
import tqdm
import torch.nn.functional as F
import time
# from fairseq2.nn.transformer.multihead_attention import AttentionWeightStoreHook
from collections import OrderedDict

from nemo.collections.asr.models import EncDecMultiTaskModel

def lang_name_3_to_2(lang_name:str):
    if lang_name == "eng":
        return "en"
    if lang_name == "fra":
        return "fr"
    if lang_name == "deu":
        return "de"
    if lang_name == "spa":
        return "es"
    return lang_name

def canary_get_log_probs_for_given_output(
    canary_model: EncDecMultiTaskModel,
    input_wave_tensor: torch.Tensor,
    target_decoder_output_token_sequence: torch.Tensor,
    src_lang: str,
    tgt_lang: str,
    task: str = None
):
    '''
    Used to get the logits for the given output tensor and input audio wave.
    Parameters:
        canary_model: EncDecMultiTaskModel
            The model to be used for inference.
        input_wave_tensor: torch.Tensor
            The input audio wave tensor with 16kHz Sample Rate.
        target_decoder_output_token_sequence: torch.Tensor
            The target decoder output token sequence.
        src_lang: str
            The source language. Can be 'en', 'es', 'de', 'fr'.
        tgt_lang: str
            The target language. Can be 'en', 'es', 'de', 'fr'.
        task: str
            The task to be performed. Can be 'transcribe' or 'translate'. If None, it will be inferred from src_lang and tgt_lang.
    '''
    canary_model.eval()
    canary_model.encoder.freeze()
    canary_model.transf_decoder.freeze()
    for param in canary_model.parameters():
        param.requires_grad = False
    original_dither_value = canary_model.preprocessor.featurizer.dither
    original_pad_to_value = canary_model.preprocessor.featurizer.pad_to

    canary_model.preprocessor.featurizer.dither = 0.0
    canary_model.preprocessor.featurizer.pad_to = 0

    input_wave_len = torch.Tensor([int(input_wave_tensor.shape[-1])]).cuda()
    canary_model.eval()
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
    tgt = torch.cat((tgt, target_decoder_output_token_sequence), dim=-1) 

    # if the target_decoder_output_token_sequence is empty, we only get the first log prob
    # so, we decode at least for one time. Note that the previous decoding is used to setting up the decoder_mems_list.
    for i in range(prompt_len, target_decoder_output_token_sequence.shape[-1] + prompt_len + 1):
        
        log_probs, decoder_mems_list = beam_search._one_step_forward(
            tgt[:, i-1:i], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i
        )
        log_probs_results.append(log_probs[0,0,:]) # we do not consider batch processing here.
        # if i == 0:
        #     print(f"log_probs: {log_probs}")

        # next_tokens = torch.argmax(log_probs[:, -1], dim=-1, keepdim=True)
        # next_tokens = beam_search.pad * pad_profile + next_tokens * (1 - pad_profile)
        # pad_profile = torch.max(pad_profile, (next_tokens == beam_search.eos).long())
        # tgt = torch.cat((tgt, next_tokens), dim=-1)

        # # abort generation if all sequences end with <eos>
        # if pad_profile.sum() == batch_size:
        #     break

    # print(tgt.dtype)
    # restore the model states
    canary_model.preprocessor.featurizer.dither = original_dither_value
    canary_model.preprocessor.featurizer.pad_to = original_pad_to_value
    return log_probs_results

def canary_predict(
    canary_model: EncDecMultiTaskModel,
    input_wave_tensor: torch.Tensor,
    src_lang: str,
    tgt_lang: str,
    task: str = None
):
    '''
    Return the translation/ transcribe result in a hypotheses list. 
    For each hypothesis, the y_sequence ended with 2, 1. (which means end of text and padding)
    Parameters:
        canary_model: EncDecMultiTaskModel
            The model to be used for inference.
        input_wave_tensor: torch.Tensor
            The input audio wave tensor with 16kHz Sample Rate.
        target_decoder_output_token_sequence: torch.Tensor
            The target decoder output token sequence.
        src_lang: str
            The source language. Can be 'en', 'es', 'de', 'fr'.
        tgt_lang: str
            The target language. Can be 'en', 'es', 'de', 'fr'.
        task: str
            The task to be performed. Can be 'transcribe' or 'translate'. If None, it will be inferred from src_lang and tgt_lang.
    '''
    canary_model.eval()
    canary_model.encoder.freeze()
    canary_model.transf_decoder.freeze()
    original_dither_value = canary_model.preprocessor.featurizer.dither
    original_pad_to_value = canary_model.preprocessor.featurizer.pad_to

    input_wave_len = torch.Tensor([int(input_wave_tensor.shape[-1])]).cuda()
    canary_model.eval()
    log_probs, encoded_len, enc_states, enc_mask = canary_model(input_signal=input_wave_tensor.cuda(), input_signal_length=input_wave_len)

    canary_model.preprocessor.featurizer.dither = 0.0
    canary_model.preprocessor.featurizer.pad_to = 0

    tokenizer = canary_model.tokenizer
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

    hypotheses = []
    beam_hypotheses = canary_model.decoding.decode_predictions_tensor(
        encoder_hidden_states=enc_states,
        encoder_input_mask=enc_mask,
        decoder_input_ids=prompt,
        return_hypotheses=True,
    )[0]  # the return value of method is [beam_hypotheses, None]
    # pdb.set_trace()

    for hypothese in beam_hypotheses:
        hypothese.text = canary_model.decoding.strip_special_tokens(hypothese.text)

    # beam_hypotheses_text = [canary_model.decoding.strip_special_tokens(hypotheses.text) for hypotheses in beam_hypotheses]
    
    hypotheses += beam_hypotheses
    # print(hypotheses)

    # restore the model states
    canary_model.preprocessor.featurizer.dither = original_dither_value
    canary_model.preprocessor.featurizer.pad_to = original_pad_to_value
    return hypotheses