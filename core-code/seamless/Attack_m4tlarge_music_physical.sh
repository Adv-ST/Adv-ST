#!/bin/bash

date
echo "job begin"


target_folder="../../../core-code/database/vctk_selected/p285" 
# Placeholder parameter, reserved for future extension. 
# Currently, it reads a raw speaker without performing any processing, and does not affect the generation of adversarial music.
# It is only used for the batch-generated adversarial music directory structure.


language_combinations=(
    "eng"
    "cmn"
    "deu"
    "fra"
    "ita"
    "spa"
)


sentences=(
        "You make me sick."
        "Shame on you."
        "Are you insane?"
        "Who do you think you're talking to?"
        "Don't waste my time anymore."
        "This is unbelievable."
        "I can't stand you."
        "This is ridiculous."
        "Stop bothering me."
        "What's wrong with you?"
)


audio_file=$(find "$target_folder" -maxdepth 1 -type f -name '*.wav' | head -n 1)
if [[ ! -z "$audio_file" ]]; then
    audio_file_name=$(basename "$audio_file" .wav)
    for lang_combo in "${language_combinations[@]}"; do
        for index in "${!sentences[@]}"; do
            sentence_index=$(($index + 1))
            sentence="${sentences[$index]}"
            lang_suffix=$(echo $lang_combo | tr ',' '-')
            out_folder="Generated/Attack-m4tlarge-(${lang_suffix})-music-physical/${audio_file_name}/${sentence_index}"
            mkdir -p "$out_folder"
            command="python Attack_seamless_music_physical.py --in \"$audio_file\" \
                    --imp ../../../core-code/AIR_regular/*.wav \
                    --target \"$sentence\" \
                    --out \"$out_folder\" \
                    --lr 0.1 \
                    --noise 0.98 \
                    --tgtl \"$lang_combo\" \
                    --iterations 5 \
                    --speech_pth ../../../core-code/LibriSpeech_wav/"
            echo "Executing: $command"
            eval "$command"
        done
    done
fi


echo "job end"
date