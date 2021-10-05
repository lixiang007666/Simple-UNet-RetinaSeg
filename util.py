import numpy as np
import cv2
import os
from PIL import Image

def read_image_and_name(path):
    imgdir = os.listdir(path)
    imglst = []
    imgs = []
    for v in imgdir:
        imglst.append(path + v)
        imgs.append(cv2.imread(path + v))
    print(imglst)
    print('original images shape: ' + str(np.array(imgs).shape))
    return imglst,imgs

def read_label_and_name(path):
    labeldir = os.listdir(path)
    labellst = []
    labels = []
    for v in labeldir:
        labellst.append(path + v)
        labels.append(np.asarray(Image.open(path + v)))
    print(labellst)
    print('original labels shape: ' + str(np.array(labels).shape))
    return labellst,labels

def resize(imgs,resize_height, resize_width):
    img_resize = []
    for file in imgs:
        img_resize.append(cv2.resize(file,(resize_height,resize_width)))
    return img_resize

#将N张576x576的图片裁剪成48x48
def crop(image,dx):
    list = []
    for i in range(image.shape[0]):
        for x in range(image.shape[1] // dx):
            for y in range(image.shape[2] // dx):
                list.append(image[ i,  y*dx : (y+1)*dx,  x*dx : (x+1)*dx]) #这里的list一共append了20x12x12=2880次所以返回的shape是(2880,48,48)
    return np.array(list)

# 网络预测输出转换成图像子块
# 网络预测输出 size=[Npatches, patch_height*patch_width, 2]
def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2  # 确认是否为二分类
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original": # 网络概率输出
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1] #pred[:, :, 0] 是反分割图像输出 pred[:, :, 1]是分割输出
    elif mode=="threshold": # 网络概率-阈值输出
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    # 输出形式改写成(Npatches,1, patch_height, patch_width)
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images
