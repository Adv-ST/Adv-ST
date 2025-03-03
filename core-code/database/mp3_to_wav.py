import os
from pydub import AudioSegment

def convert_ogg_to_wav(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为 .ogg 文件
        if filename.endswith(".mp3"):
            # 构建文件的完整路径
            ogg_path = os.path.join(directory, filename)
            # 更改文件扩展名为 .wav
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(directory, wav_filename)
            
            # 读取 .ogg 文件
            audio = AudioSegment.from_file(ogg_path, format="mp3")
            # 导出为 .wav 文件
            audio.export(wav_path, format="wav")
            
            print(f"转换完成：{ogg_path} -> {wav_path}")

# 设置文件夹路径
folder_path1 = "de/de_1"
folder_path2 = "fr/fr_1" 
folder_path3 = "it/it_1" 
folder_path1_1 = "de/de_2"
folder_path2_2 = "fr/fr_2" 
folder_path3_3 = "it/it_2" 
convert_ogg_to_wav(folder_path1)
convert_ogg_to_wav(folder_path2)
convert_ogg_to_wav(folder_path3)
convert_ogg_to_wav(folder_path1_1)
convert_ogg_to_wav(folder_path2_2)
convert_ogg_to_wav(folder_path3_3)
