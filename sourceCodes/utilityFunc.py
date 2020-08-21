import cv2
import numpy as np
import glob
import os
import math
from skimage.measure import compare_ssim as ssim


def resizeAndSave(data_path, save_path, resize):
    data_path = glob.glob(data_path)
    os.makedirs(save_path)
    for i in range(len(data_path)):
        fn = data_path[i].split('\\')[-1]
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        LRImg = cv2.resize(img, (resize, resize), cv2.INTER_AREA)
        cv2.imwrite(save_path + "{}".format(str(fn)), LRImg)
        print("{}. dosya kaydediliyor.".format(str(i)))
    print("İşlem bitti.")


def savingLR(data_path, save_path, factor):
    data_path = glob.glob(data_path)
    os.makedirs(save_path)
    for i in range(len(data_path)):
        fn = data_path[i].split('\\')[-1]
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        width = img.shape[1]
        height = img.shape[0]
        newWidth = int(width / factor)
        newHeight = int(height / factor)
        img = cv2.resize(img, (newWidth, newHeight), cv2.INTER_CUBIC)
        LRImg = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
        cv2.imwrite(save_path + "{}".format(str(fn)), LRImg)
        print("{}. dosya kaydediliyor.".format(str(i)))
    print("İşlem bitti.")


def cropWithStride(data_path, save_path):
    data_path = glob.glob(data_path)
    os.makedirs(save_path)
    for i in range(len(data_path)):
        fn = data_path[i].split('\\')[-1].split('.')[0]
        stride = 14
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        print("{}. resim kırpılıyor.".format(str(i)))
        for i in range(0, 1024 - 32, stride):
            for j in range(0, 1024 - 32, stride):
                crop_img =  img[j:j + 32, i:i + 32].copy()
                cv2.imwrite(save_path + "{}_{}.png".format(str(fn), str(j)), crop_img)
    print("İşlem bitti.")


def crop2(data_path, save_path, crop_size, totalIterNumber):
    data_path = glob.glob(data_path)
    os.makedirs(save_path)
    for i in range(len(data_path)):
        fn = data_path[i].split('\\')[-1].split('.')[0]
        #print(fn)
        img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
        print("{}. resim kırpılıyor.".format(str(i)))
        for j in range(totalIterNumber):
            for x in range(totalIterNumber):
                crop_img =  img[j*crop_size:(j+1)*crop_size, x*crop_size:(x+1)*crop_size].copy()
                print("{}_{}_{}.png kaydediliyor...".format(str(fn), str(j), str(x)))
                cv2.imwrite(save_path + "{}_{}_{}.png".format(str(fn), str(j), str(x)), crop_img)
        
    print("İşlem bitti!")


def convertLR(data_path, OR_NAME, IMG_NAME, factor, size):
    img = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), cv2.INTER_AREA)
    cv2.imwrite(OR_NAME, img)
    width = img.shape[1]
    height = img.shape[0]
    newWidth = int(width / factor)
    newHeight = int(height / factor)
    img = cv2.resize(img, (newWidth, newHeight), cv2.INTER_CUBIC)
    LRImg = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
    cv2.imwrite(IMG_NAME, LRImg)
    LRImg_YCrCb = cv2.cvtColor(LRImg, cv2.COLOR_RGB2YCrCb)
    y = LRImg_YCrCb[:, :, 0]
    cb = LRImg_YCrCb[:, :, 1]
    cr = LRImg_YCrCb[:, :, 2]
    y = y.astype(float) / 255.
    print(y.shape)
    return y, cb, cr

def convertTestLR(data_path, factor, size):
    img = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), cv2.INTER_AREA)
    orijinalResim = img
    width = img.shape[1]
    height = img.shape[0]
    newWidth = int(width / factor)
    newHeight = int(height / factor)
    img = cv2.resize(img, (newWidth, newHeight), cv2.INTER_CUBIC)
    LRImg = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
    bozukResim = LRImg
    LRImg_YCrCb = cv2.cvtColor(LRImg, cv2.COLOR_RGB2YCrCb)
    y = LRImg_YCrCb[:, :, 0]
    cb = LRImg_YCrCb[:, :, 1]
    cr = LRImg_YCrCb[:, :, 2]
    y = y.astype(float) / 255.
    return y, cb, cr, orijinalResim, bozukResim

# define a function for peak signal-to-noise ratio (PSNR)
def PSNR(changedImg, originalImg):
    #changedImg = np.array(changedImg)     
    changedImg = changedImg.astype(float)
    originalImg = originalImg.astype(float)

    diff = originalImg - changedImg
    diff = diff.flatten('C')

    RMSE = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / RMSE)


# define function for mean squared error (MSE)
def MSE(changedImg, originalImg):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((changedImg.astype('float') - originalImg.astype('float')) ** 2)
    err /= float(changedImg.shape[0] * changedImg.shape[1])
    return err


def prepareData(data_path, label_path):
  data_path = glob.glob(data_path)
  label_path = glob.glob(label_path)
  for i in range(len(data_path)):
    img = cv2.imread(data_path[i], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path[i], cv2.IMREAD_UNCHANGED)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    label_YCrCb = cv2.cvtColor(label, cv2.COLOR_RGB2YCrCb)
    img_YCrCb = (img_YCrCb).astype(float) / 255.
    label_YCrCb = (label_YCrCb).astype(float) / 255.
    img_y = img_YCrCb[:, :, 0]
    label_y = label_YCrCb[:, :, 0]
  return img_y, label_y
