import os
import shutil
import cv2
import dlib
folder_path = r"/home/featurize/DualStyleGAN/data/vox1_1s/vox1_s/train/"


# 加载预训练的面部检测和形状预测模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '/home/featurize/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')

for file_name in os.listdir(folder_path):
    # 拼接文件的完整路径
    file_path = os.path.join(folder_path, file_name) #\vox1_s\train\id10001#DtdEYdViWdw#000066#000172
    # 判断是否为文件
    print(file_name)
    for file_name_jpg in os.listdir(file_path):
        file_path_jpg = os.path.join(file_path, file_name_jpg)  #\vox1_s\train\id10001#DtdEYdViWdw#000066#000172\0000000.png
        if file_path_jpg.endswith(".jpg") or file_path_jpg.endswith(".png"):

            image = cv2.imread(file_path_jpg)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if not faces:
                print(file_path_jpg)
                os.remove(file_path_jpg)
            if len(faces) != 1:
                print(file_path_jpg)
                os.remove(file_path_jpg)


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



