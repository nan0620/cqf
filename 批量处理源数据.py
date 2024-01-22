import os
import shutil

# 源文件夹和目标文件夹
source_folder = '/Users/nanjiang/cqf/spx_eod_2023'
destination_folder = '/Users/nanjiang/cqf/spx_eod_unhandled'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 文件的完整路径
    source_file = os.path.join(source_folder, filename)

    # 如果是文件而非文件夹
    if os.path.isfile(source_file):
        # 目标文件的完整路径
        destination_file = os.path.join(destination_folder, filename)

        # 复制文件
        shutil.copy(source_file, destination_file)
        print(f'文件 {filename} 已被复制到 {destination_folder}')
