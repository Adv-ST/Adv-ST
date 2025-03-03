#!/bin/bash

date
echo "job begin"

# Speaker dir
target_folders=(
    "../../core-code/database/vctk_selected/p285"
    "../../core-code/database/vctk_selected/p314"
    "../../core-code/database/aishell_selected/SSB0005"
    "../../core-code/database/aishell_selected/SSB1408"
    "../../core-code/database/voxpopuli_es/2009"
    "../../core-code/database/voxpopuli_es/2011"
    "../../core-code/database/de/de_1"
    "../../core-code/database/de/de_2"
    "../../core-code/database/fr/fr_1"
    "../../core-code/database/fr/fr_2"
    "../../core-code/database/it/it_1"
    "../../core-code/database/it/it_2"
)

# attack languages
language_combinations=(
    "eng,cmn,deu,fra"
)

# target semantic
sentences=(
    "You make me sick."
    "Shame on you."
    "Are you insane?"
)

epsilons=(0.5 0.1 0.01)

for target_folder in "${target_folders[@]}"; do
    audio_file=$(find "$target_folder" -maxdepth 1 -type f -name '*.wav' | head -n 1)
    if [[ ! -z "$audio_file" ]]; then
        audio_file_name=$(basename "$audio_file" .wav)
        for epsilon in "${epsilons[@]}"; do
            for lang_combo in "${language_combinations[@]}"; do
                for index in "${!sentences[@]}"; do
                    sentence_index=$(($index + 1))
                    sentence="${sentences[$index]}"
                    lang_suffix=$(echo $lang_combo | tr ',' '-')
                    out_folder="Generated/Attack-m4tlarge-eps-(${epsilon})-(${lang_suffix})-cycle/${audio_file_name}/${sentence_index}"
                    mkdir -p "$out_folder"
                    command="python Attack_seamless.py --in \"$audio_file\" \
                            --target \"$sentence\" \
                            --out \"$out_folder\" \
                            --lr 0.1 \
                            --eps ${epsilon} \
                            --bp 1 \
                            --tgtl \"$lang_combo\" \
                            --tgtm \"seamlessM4T_large\" \
                            --cycle 1 "
                    echo "Executing: $command"
                    eval "$command"
                done
            done
        done
    fi
done

echo "job end"
date