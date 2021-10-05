import cv2
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import model_from_json

from unet import get_UNet
from util import pred_to_imgs
import matplotlib.pyplot as plt


if __name__ == '__main__':

    resize_height, resize_width = (576, 576)
    dx = 48
    #读取预测图片
    imgs = cv2.imread('DRIVE/test/images/01_test.tif')[...,1] #读取G通道
    imgs = np.array(cv2.resize(imgs,(resize_height,resize_width))) #imgs现在是576x576大小
    #读取预测图片的标签
    label = np.array(Image.open('DRIVE/test/1st_manual/01_manual1.gif'))
    #预测图片和标签标准化
    X_test = imgs.astype('float32')/255
    print('X_test original shape: '+str(X_test.shape))
    Y_test = label.astype('float32')/255


    #对预测图片进行裁剪按行优先，裁剪成(144,48,48)
    list = []
    for i in range(resize_height//dx):
        for j in range(resize_width//dx):
            list.append(X_test[i*dx:(i+1)*dx, j*dx:(j+1)*dx])
    X_test = np.array(list)[:,:,:,np.newaxis,...] #增加一维变成(144,1,48,48)
    print('input shape: '+str(X_test.shape))

    #加载模型和权重并预测
    with open(r'model_architecture.json', 'r') as file:
        model_json1 = file.read()
    model = model_from_json(model_json1)
    model.load_weights('best_weights.h5')
    Y_pred = model.predict(X_test)
    print('predict shape: '+str(Y_pred.shape)) #预测结果的shape是(Npatches,patch_height*patch_width,2)

    #把预测输出的numpy数组拼接还原再显示
    Y_pred = Y_pred[..., 0]  #二分类提取出分割前景 现在Y_pred的shape是(144,2304) 且这个144是按照行优先来拼接的

    #对预测结果进行拼接，将(144,2304)拼接成(576,576)
    t=0
    image = np.zeros((resize_height,resize_width))
    for i in range(resize_height//dx):
        for j in range(resize_width//dx):
            temp = Y_pred[t].reshape(dx,dx)
            image[i*dx:(i+1)*dx, j*dx:(j+1)*dx] = temp
            t = t+1
    image = cv2.resize(image,((Y_test.shape[1], Y_test.shape[0]))) #将576x576大小的图像还原成原图像大小
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.figure(figsize=(6, 6))
    plt.imshow(Y_test)
    plt.show()
