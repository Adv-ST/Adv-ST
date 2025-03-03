import os
import librosa
import soundfile

def resample_wav_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.wav'):
            print("Processing:", filepath)
            y, sr = librosa.load(filepath, sr=None)
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
            soundfile.write(filepath, y_resampled, 16000)
            print("Resampling completed for:", filepath)

directory = "./downloads"

# 调用函数
resample_wav_files(directory)
