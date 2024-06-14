import os
import shutil
folder_path = r"/home/featurize/DualStyleGAN/data/vox1_1s/vox1_s/train/"
def delete_empty_folders(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image_count = 0
        for file_name_jpg in os.listdir(file_path):
            file_path_jpg = os.path.join(file_path, file_name_jpg)
            if file_path_jpg.endswith(".jpg") or file_path_jpg.endswith(".png"):
                image_count += 1
        if image_count < 8:
            #os.rmdir(file_path)
            shutil.rmtree(file_path)
            print(f"Deleted empty folder: {file_name}")

# 调用函数删除空文件夹
delete_empty_folders(folder_path)