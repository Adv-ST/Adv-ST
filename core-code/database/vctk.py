import os
import shutil

def copy_first_audio(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for root, dirs, files in os.walk(src_folder):
        for d in dirs:
            subfolder_path = os.path.join(root, d)
            audio_files = [f for f in os.listdir(subfolder_path) if f.endswith('.flac')]
            if audio_files:
                first_audio = audio_files[0]
                src_path = os.path.join(subfolder_path, first_audio)
                
                relative_path = os.path.relpath(subfolder_path, src_folder)
                target_subfolder_path = os.path.join(dst_folder, relative_path)
                if not os.path.exists(target_subfolder_path):
                    os.makedirs(target_subfolder_path)
                
                shutil.copy(src_path, target_subfolder_path)
                print(f"{src_path}\t-->\t{target_subfolder_path}")

# 使用示例
src_folder = '~/data/VCTK/wav48_silence_trimmed'
dst_folder = 'vctk_selected'
copy_first_audio(src_folder, dst_folder)









import os
import subprocess

def convert_flac_to_wav(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.flac'):
                # 构建完整的文件路径
                flac_file_path = os.path.join(root, file)
                # 创建对应的.wav文件路径
                wav_file_path = flac_file_path.rsplit('.', 1)[0] + '.wav'
                # 使用ffmpeg进行转换
                cmd = f'ffmpeg -i "{flac_file_path}" "{wav_file_path}"'
                subprocess.run(cmd, shell=True)
                print(f'Converted: {flac_file_path} to {wav_file_path}')

# 使用示例
convert_flac_to_wav(dst_folder)
