

import dlib
import cv2
import matplotlib.pyplot as plt

# 加载预训练的面部检测和形状预测模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/featurize/DualStyleGAN/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')
# cv2读取图片
image = cv2.imread("/home/featurize/DualStyleGAN/data/SCUT-FBP5500_v2/imagess/AM1000.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测面部
faces = detector(gray)
print(faces)
# 绘制面部矩形框和关键点
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 预测面部关键点
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

# 使用Matplotlib显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
plt.show()
plt.axis('off')  # 关闭坐标轴
plt.show()
