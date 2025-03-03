# Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks

This repository contains the source code for the research paper **"Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks"**. Visit our [project website](https://adv-st.github.io) to explore audio samples and learn more.

- [Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks](#exploiting-vulnerabilities-in-speech-translation-systems-through-targeted-adversarial-attacks)
  - [Getting Started](#getting-started)
  - [Attack Seamless](#attack-seamless)
    - [Perturbation-based Attack](#perturbation-based-attack)
      - [Environment Setup](#environment-setup)
      - [Pretrained Models](#pretrained-models)
      - [Run Attack](#run-attack)
    - [Music-based Attack](#music-based-attack)
      - [Environment Setup](#environment-setup-1)
      - [Run Attack](#run-attack-1)
    - [Physical attack](#physical-attack)
      - [Environment Setup](#environment-setup-2)
      - [Datasets](#datasets)
      - [Run Attack](#run-attack-2)
    - [Evaluation](#evaluation)
      - [Environment Setup](#environment-setup-3)
      - [Pretrained Models](#pretrained-models-1)
      - [Run Evaluation](#run-evaluation)
  - [Attack Canary](#attack-canary)
    - [Perturbation-based Attack](#perturbation-based-attack-1)
      - [Environment Setup](#environment-setup-4)
      - [Pretrained Models](#pretrained-models-2)
      - [Generate reference audio](#generate-reference-audio)
      - [Run Attack](#run-attack-3)
    - [Music-based Attack](#music-based-attack-1)
      - [Environment Setup](#environment-setup-5)
      - [Run Attack](#run-attack-4)
    - [Physical attack](#physical-attack-1)
      - [Environment Setup](#environment-setup-6)
      - [Datasets](#datasets-1)
      - [Run Attack](#run-attack-5)
    - [Evaluation](#evaluation-1)
      - [Environment Setup](#environment-setup-7)
      - [Run Evaluation](#run-evaluation-1)
  - [Acknowledgements](#acknowledgements)
      - [Open-Source Projects](#open-source-projects)
      - [Datasets](#datasets-2)

## Getting Started

The experiments were conducted on NVIDIA GeForce A6000, The code is written in Python. We recommend using a virtual environment (e.g., conda) to manage dependencies.






## Attack Seamless
### Perturbation-based Attack
#### Environment Setup
1. **Clone the Seamless Repository:**

   ```bash
      cd our_repository
      git clone https://github.com/facebookresearch/seamless_communication.git
      cd seamless_communication
      git checkout a9f6fa2c98f93af0ff1a9d967424a85b8fd352f1
      conda create -n advst python=3.9.18
      conda activate advst
      pip install .
   ```
2. **Install Dependencies:**
    ```bash
      conda install -c conda-forge libsndfile==1.0.31  # Not available via pip
    ```

#### Pretrained Models  

We evaluated the following Seamless models in our experiments:  
- **SeamlessM4T Large**  
- **SeamlessM4T Medium**  
- **SeamlessM4T v2**  
- **Seamless Expressive**  

Details on these models and download instructions can be found in the [official repository](https://github.com/facebookresearch/seamless_communication). Notably, we use the Seamless Expressive model to evaluate VSIM-E. Access to the pretrained Expressive model requires official authorization from Meta. For more information, refer to the [official model repository](https://huggingface.co/facebook/seamless-expressive).  



#### Run Attack
- For batch attack:

   ```bash
      cp core-code/seamless/{Attack_seamless.py,psy.py,Attack_m4tlarge.sh,Attack_m4tmedium.sh,Attack_m4tv2.sh,Attack_expressive.sh} seamless_communication/src/

      cd seamless_communication/src/
      # conda env: advst
      bash Attack_m4tlarge.sh
      bash Attack_m4tmedium.sh
      bash Attack_m4tv2.sh
      bash Attack_expressive.sh
   ```

- Attack with Target Cycle Optimization:
   ```bash
      cp core-code/seamless/{Attack_m4tlarge_tco.sh} seamless_communication/src/
      
      cd seamless_communication/src/
      # conda env: advst
      bash Attack_m4tlarge_tco.sh
   ```

- For more convient attack test: single sample attack
   ```bash
      # conda env: advst
      python Attack_seamless.py  --in audio_file \
                                 --target \"You make me sick.\" \
                                 --out \"Attack-m4tlarge-(eng,deu,fra,cmn)/${speaker}/${sentence_index}\" \
                                 --lr 0.1 \
                                 --eps 0.5 \
                                 --bp 1 \
                                 --tgtl \"eng,cmn,deu,fra\""
   ```
























---
### Music-based Attack

#### Environment Setup
1. Setup seamless env as mentioned in perturbation based attack.
2. Setup MusTango env
   -  Clone the modified Mustango model code. We have modified the [official MusTango code](https://github.com/AMAAI-Lab/mustango) to remove the `no_grad` operation during the music generation process, ensuring that the adversarial music optimization process enables gradient flow.

      ```bash
         cp -r core-code/seamless/mustango seamless_communication/src/
      ```
   -  Install Dependencies
      ```bash
         # conda env: advst
         pip install -r core-code/music_req.txt

         cd seamless_communication/src/mustango/diffusers
         pip install .
      ```

#### Run Attack
- For batch attack:

   ```bash
      cp core-code/seamless/{Attack_seamless_music.py,Attack_m4tlarge_music.sh,Attack_m4tmedium_music.sh,Attack_m4tv2_music.sh,Attack_expressive_music.sh} seamless_communication/src/mustango/

      cd seamless_communication/src/mustango/
      # conda env: advst
      bash Attack_m4tlarge_music.sh
      bash Attack_m4tmedium_music.sh
      bash Attack_m4tv2_music.sh
      bash Attack_expressive_music.sh
   ```

- Attack with Target Cycle Optimization:
   ```bash
      cp core-code/seamless/{Attack_m4tlarge_music_tco.sh} seamless_communication/src/mustango/
      
      cd seamless_communication/src/mustango/
      bash Attack_m4tlarge_music_tco.sh
   ```

- For more convient attack test: single sample attack
   ```bash
      # conda env: advst
      python Attack_seamless_music.py     --target \"You make me sick.\" \
                                          --out \"Attack-m4tlarge-(eng,deu,fra,cmn)-music/${speaker}/${sentence_index}\" \
                                          --lr 0.1 \
                                          --tgtl \"eng,cmn,deu,fra\""
   ```




---
### Physical attack
#### Environment Setup
- Setup seamless and Mustango env as mentioned in music based attack.

#### Datasets
1. We incorporate the [Aachen Impulse Response Database](https://openslr.org/20/) to simulate environmental reverberation.
   - [Aachen Impulse Response Database](https://openslr.org/20/)
       - Manual download
       - Get the .wav file versions (AIR_wav_files.zip)

2. We use the [LibriSpeech](https://www.openslr.org/12) dataset to simulate real-world background voice, enhancing the adversarial music through adversarial augmentation during optimization.
   - [LibriSpeech](https://www.openslr.org/12)
      - Splits: `train-clean-100`
         ```bash
            cd core-code
            wget https://openslr.elda.org/resources/12/train-clean-100.tar.gz
            tar -zxvf train-clean-100.tar.gz
            python move_flac_to_wav.py
         ```

#### Run Attack
- For batch attack:

   ```bash
      cp core-code/seamless/{Attack_seamless_music_physical.py,dataset.py,Attack_m4tlarge_music_physical.sh} seamless_communication/src/mustango/

      cd seamless_communication/src/mustango/
      # conda env: advst
      bash Attack_m4tlarge_music_physical.sh
   ```

- For more convient attack test: single sample attack
   ```bash
      # conda env: advst
      python Attack_seamless_music_physical.py     --target \"You make me sick.\" \
                                                   --out \"Attack-m4tlarge-(eng,deu,fra,cmn)-music/${speaker}/${sentence_index}\" \
                                                   --lr 0.1 \
                                                   --noise 0.98 \
                                                   --tgtl \"eng\" \
                                                   --speech_pth \"../../../core-code/LibriSpeech_wav/\"
   ```




---
### Evaluation
#### Environment Setup
Due to dependency conflicts between the music generation model and the language model used for testing semantic similarity, we need to create a new virtual isolated environment for evaluation.
```bash
   cd seamless_communication
   conda create -n adv-seamless-eval python=3.9.18
   conda activate adv-seamless-eval
   pip install .

   pip install -r ../core-code/seamless/eval_req.txt

   conda install -c conda-forge libsndfile==1.0.31
```

#### Pretrained Models
We use the Seamless Expressive model to evaluate VSIM-E. Notably, access to the pretrained Expressive model requires official authorization from Meta. For details, refer to the [official model repository](https://huggingface.co/facebook/seamless-expressive).



#### Run Evaluation
```bash
   cp core-code/seamless/{seamless_evaluate.py} seamless_communication/src/

   cd seamless_communication/src/
   # conda env: adv-seamless-eval
   python seamless_evaluate.py      --tgtm "seamlessM4T_large" \
                                    --mode "music" \
                                    --in "../../demo_samples/seamless/mustango/Generated/Attack-m4tlarge-(eng-cmn-deu)-music/p285_126_mic1/3/cqm-adv-409.wav"  \
                                    --target_lang "cmn"  \
                                    --target_semantic "Are you insane?" \
                                    --gated_model_dir "./facebook/seamless-expressive" # the downloaded seamless expressive model dir

   
   python seamless_evaluate.py      --tgtm "seamlessM4T_large" \
                                    --mode "perturbation" \
                                    --in "../../demo_samples/seamless/Generated/Attack-m4tlarge-eps-(0.1)-(eng-cmn-deu-fra)/p285_126_mic1/5/mrv-adv-2565.wav"  \
                                    --speaker_lang "eng"  \
                                    --target_lang "cmn"  \
                                    --target_semantic "Don't waste my time anymore." \
                                    --original_audio "../../core-code/database/vctk_selected/p285/p285_126_mic1.wav" \
                                    --gated_model_dir "./facebook/seamless-expressive" # the downloaded seamless expressive model dir
```



































































---
## Attack Canary
### Perturbation-based Attack
#### Environment Setup
1. **Clone the Modified NeMo Module:**
   We have modified the [official NeMo](https://github.com/NVIDIA/NeMo/tree/v1.23.0) code to remove the **no_grad** operation, ensuring that the adversarial optimization process enables gradient flow.
   ```bash
      cp -r core-code/canary/NeMo-1.23.0 ./
   ```

2. **Setup Virtual Envirement:**
   As canary uses different base dependencies from seamless, we need build a new virtual python envirement for attack canary.
   ```bash
      conda create -n advst-canary python=3.10.12
      conda activate advst-canary
      pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]
      conda install -c conda-forge libsndfile==1.0.31  # Not available via pip
      pip install transformers==4.41.2
      pip install datasets==2.20.0
      pip install huggingface-hub==0.23.4
   ```


#### Pretrained Models
We evaluated the **canary-1b** in our experiments. Model details can be found in their [official repository](https://huggingface.co/nvidia/canary-1b).


#### Generate reference audio
- Unlike Seamless, Canary can only accept speech as input. Therefore, we need to generate a speech sample for each attack target semantic (in English) to generate the corresponding text for the target semantic in the specified language, which will be stored in "../core-code/canary/reference".
- We use Seamless to generate the target speech. **The semantics used in the experiments have already been generated.** If you need to perform attacks on additional target semantics, you can generate the corresponding speech using the following command:
   ```bash
   # conda env: advst
   # m4t_predict {target semantic (in eng)} --task t2st --tgt_lang "eng" --src_lang "eng" --output_path "../core-code/canary/reference/{target semantic (in eng)}"
   m4t_predict "This is ridiculous." --task t2st --tgt_lang "eng" --src_lang "eng" --output_path "../core-code/canary/reference/This is ridiculous..wav"
   ```



#### Run Attack
- For batch attack:
   ```bash
      cp core-code/canary/{Attack_canary.py,psy.py,Attack_canary.sh} NeMo-1.23.0/

      cd NeMo-1.23.0/
      # conda env: advst-canary
      bash Attack_canary.sh
   ```

- For more convient attack test: single sample attack
   ```bash
      # conda env: advst-canary
      python Attack_canary.py  --in audio_file \
                                 --target \"You make me sick.\" \
                                 --out \"Generated/Attack-canary-eps-(0.5)-(eng,fra,deu,spa)/${speaker}/${sentence_index}\" \
                                 --lr 0.1 \
                                 --eps 0.5 \
                                 --bp 1 \
                                 --src_lang \"eng\" \
                                 --tgtl \"eng,fra,deu,spa\""
   ```



---
### Music-based Attack

#### Environment Setup
1. Setup canary env as mentioned in perturbation based attack.
2. Setup MusTango env
   -  Clone the modified Mustango model code. We have modified the [official MusTango code](https://github.com/AMAAI-Lab/mustango) to remove the `no_grad` operation during the music generation process, ensuring that the adversarial music optimization process enables gradient flow.

      ```bash
         cp -r core-code/mustango NeMo-1.23.0/
      ```
   -  Install Dependencies
      ```bash
         # conda env: advst-canary
         pip install -r core-code/music_req.txt

         cd NeMo-1.23.0/mustango/diffusers
         pip install .
         pip install torchaudio
      ```

#### Run Attack
- For batch attack:

   ```bash
      cp core-code/canary/{Attack_canary_music.py,Attack_canary_music.sh} NeMo-1.23.0/

      cd NeMo-1.23.0/mustango/
      # conda env: advst-canary
      bash Attack_canary_music.sh
   ```

- For more convient attack test: single sample attack
   ```bash
      # conda env: advst-canary
      python Attack_canary_music.py       --target \"You make me sick.\" \
                                          --out \"Generated/Attack-canary-(eng,fra,deu,spa)-music/${speaker}/${sentence_index}\" \
                                          --lr 0.1 \
                                          --tgtl \"eng,fra,deu,spa\""
   ```



---
### Physical attack
#### Environment Setup
- Setup Canary and Mustango env as mentioned in music based attack.

#### Datasets
- Set up the datasets as outlined in the Physical Attack on Seamless.

#### Run Attack
- For batch attack:

   ```bash
      cp core-code/canary/{Attack_canary_music_physical.py,Attack_canary_music_physical.sh,dataset.py} NeMo-1.23.0/mustango/

      cd NeMo-1.23.0/mustango/
      # conda env: advst-canary
      bash Attack_canary_music_physical.sh
   ```

- For more convient attack test: single sample attack
   ```bash
      # conda env: advst-canary
      python Attack_seamless_music_physical.py     --target \"You make me sick.\" \
                                                   --out \"Attack-m4tlarge-(eng,deu,fra,cmn)-music/${speaker}/${sentence_index}\" \
                                                   --lr 0.1 \
                                                   --noise 0.98 \
                                                   --tgtl \"eng\""
   ```


---
### Evaluation
#### Environment Setup
Due to dependency conflicts between the music generation model and the language model used for testing semantic similarity, we need to create a new virtual isolated environment for evaluation.
1. **Install Seamless env for VSIM-E calculate:**
   ```bash
      cd seamless_communication
      conda create -n adv-canary-eval python=3.10.12
      conda activate adv-canary-eval
      pip install .
   ```

2. **Install NeMo env:**
   ```bash
      cd ../NeMo-1.23.0/
      pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]

      # For the bert model of evaluation 
      conda install -c conda-forge libsndfile==1.0.31  # Not available via pip
      pip install -r ../core-code/canary/eval_req.txt
   ```

#### Run Evaluation
```bash
   cp core-code/canary/{canary_evaluate.py} NeMo-1.23.0/

   cd NeMo-1.23.0/
   # conda env: adv-canary-eval
   python canary_evaluate.py        --mode "music" \
                                    --in "../../demo_samples/canary/mustango/Generated/Attack-canary-(eng-fra-deu)-music/p285_126_mic1/3/zws-adv-580.wav"  \
                                    --target_lang "fra"  \
                                    --target_semantic "Are you insane?" \
                                    --gated_model_dir "./facebook/seamless-expressive" # the downloaded seamless expressive model dir


   python canary_evaluate.py        --mode "perturbation" \
                                    --in "../../demo_samples/canary/Generated/Attack-canary-eps-(0.1)-(eng-fra-deu)/p285_126_mic1/4/bcb-adv-959.wav"  \
                                    --speaker_lang "eng"  \
                                    --target_lang "spa"  \
                                    --target_semantic "Who do you think you're talking to?" \
                                    --original_audio "../core-code/database/vctk_selected/p285/p285_126_mic1.wav" \
                                    --gated_model_dir "./facebook/seamless-expressive" # the downloaded seamless expressive model dir
```


---
## Acknowledgements
We sincerely appreciate the invaluable contributions of the authors and developers behind the open-source projects and datasets utilized in this research. Their dedication to advancing scientific knowledge and fostering open collaboration has been instrumental in making this work possible. We acknowledge their efforts in providing high-quality resources that have significantly contributed to the progress of our study.  

#### Open-Source Projects  
- **[Seamless Communication](https://github.com/facebookresearch/seamless_communication)**
- **[Canary](https://github.com/NVIDIA/NeMo/tree/v1.23.0)**
- **[MusTango](https://github.com/AMAAI-Lab/mustango)**

#### Datasets  
- **[LibriSpeech](https://www.openslr.org/12)** 
- **[Aachen Impulse Response Database](https://openslr.org/20/)**
- **[Common Voice](https://commonvoice.mozilla.org/en/datasets)**  
- **[VCTK](https://datashare.ed.ac.uk/handle/10283/2651)**
- **[VoxPopuli](https://github.com/facebookresearch/voxpopuli)**
- **[AISHELL](https://openslr.org/33/)**

We deeply appreciate the open-source community’s commitment to sharing knowledge and resources, which continues to drive innovation in machine learning and security.