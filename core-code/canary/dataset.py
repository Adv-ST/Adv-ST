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
        # wav = wav[:,:self.audio_len]
        # 计算需要填充的长度
        padding = self.audio_len - wav.size(-1)
        # 填充delta_filtered以使其与original的长度相同
        # wav = F.pad(wav, (0, padding))
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
        # 遍历目录中的所有文件和子目录
        for root, dirs, files in os.walk(self.dataset_path):
            # 遍历当前目录下的所有文件
            for file in files:
                # 如果文件是wav格式，则将其路径添加到列表中
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
        # 设置随机截取的长度，范围在3到5秒之间
        min_length = int(3 * self.sample_rate)
        max_length = int(5 * self.sample_rate)
        target_length = random.randint(min_length, max_length)

        # 如果音频长度大于目标长度，随机选择起始点进行截取
        if wav.size(-1) > target_length:
            start = random.randint(0, wav.size(-1) - target_length)
            wav = wav[:, start:start + target_length]
        
        # 如果音频长度小于目标长度，进行填充
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
        # 遍历目录中的所有文件和子目录
        for root, dirs, files in os.walk(self.dataset_path):
            # 遍历当前目录下的所有文件
            for file in files:
                # 如果文件是wav格式，则将其路径添加到列表中
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files