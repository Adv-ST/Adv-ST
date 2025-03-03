from concurrent.futures import process
import os
import torch
# import julius
import torchaudio
from torch.utils.data import Dataset
import random
import librosa
import torch.nn.functional as F
import pdb
import glob

class wav_dataset_librosa(Dataset):
    def __init__(self, raw_dataset_path, flag='train', leng=16000*2):
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.wavs = self.process_meta()
        # pdb.set_trace()
        self.audio_len = int(leng)
        self.sample_list = []
        self.sample_rate = 16000
        # for idx in range(len(self.wavs)):
        #     audio_name = self.wavs[idx]
        #     wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
        #     wav = torch.Tensor(wav).unsqueeze(0)
        #     wav = wav[:,:self.audio_len]
        #     sample = {
        #         "matrix": wav,
        #         "sample_rate": sr2,
        #         "patch_num": 0,
        #         "pad_num": 0,
        #         "name": audio_name
        #     }
        #     self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
        wav = torch.Tensor(wav).unsqueeze(0)

        sample = {
            "matrix": wav.squeeze(0),
            "sample_rate": sr2,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        wav_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files
    
    
class wav_dataset_librosa_cut(Dataset):
    def __init__(self, raw_dataset_path, flag='train', leng=16000*2):
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.wavs = self.process_meta()
        # pdb.set_trace()
        self.audio_len = int(leng)
        self.sample_list = []
        self.sample_rate = 16000
        # for idx in range(len(self.wavs)):
        #     audio_name = self.wavs[idx]
        #     wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
        #     wav = torch.Tensor(wav).unsqueeze(0)
        #     wav = wav[:,:self.audio_len]
        #     sample = {
        #         "matrix": wav,
        #         "sample_rate": sr2,
        #         "patch_num": 0,
        #         "pad_num": 0,
        #         "name": audio_name
        #     }
        #     self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
        wav = torch.Tensor(wav).unsqueeze(0)
        min_length = int(3 * self.sample_rate)
        max_length = int(5 * self.sample_rate)
        target_length = random.randint(min_length, max_length)

        if wav.size(-1) > target_length:
            start = random.randint(0, wav.size(-1) - target_length)
            wav = wav[:, start:start + target_length]
        
        padding = target_length - wav.size(-1)
        if padding > 0:
            wav = F.pad(wav, (0, padding))
        sample = {
            "matrix": wav.squeeze(0),
            "sample_rate": sr2,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        wav_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files