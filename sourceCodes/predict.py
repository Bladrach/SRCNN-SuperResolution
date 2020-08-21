import torch
import numpy as np
import model
import cv2
import utilityFunc
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt


net = model.Net()

net.load_state_dict(torch.load(".\\saved_models_CROP64_51200_2\\net_epoch_120.pth"))
size = 512                                                                                                 
y, cb, cr = utilityFunc.convertLR(r"C:\Users\Mehmet\Desktop\HighResPyTorch\orijinalResimler\label\test_label\0100.png", 
                                    '{}x{}OrijResim.png'.format(size, size), 'bulanikResim.png', factor = 2, size = size)

y = torch.from_numpy(y).view(1, 1, y.shape[1], y.shape[0])
print(y.shape)
pre = net(y)
print(pre.shape)
pre = (pre.squeeze(0)).permute(1, 2, 0) 
pre = pre.detach().numpy() * 255.
print(pre.shape)
pre[pre[:] > 255] = 255
pre[pre[:] < 0] = 0
pre = pre.astype(np.uint8)
pre = pre[:,:, 0]
print(pre.shape)
print(cb.shape)
print(cr.shape)

img = cv2.merge([pre, cb, cr])

img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
cv2.imwrite("SRtahmin.png", img)

orijinal = cv2.imread("C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\{}x{}OrijResim.png".format(size, size), cv2.IMREAD_COLOR)
SRCNN = cv2.imread(r"C:\Users\Mehmet\Desktop\HighResPyTorch\SRtahmin.png", cv2.IMREAD_COLOR)
bicubic = cv2.imread(r"C:\Users\Mehmet\Desktop\HighResPyTorch\bulanikResim.png", cv2.IMREAD_COLOR)
psnrCubic = utilityFunc.PSNR(bicubic, orijinal)
psnrSR = utilityFunc.PSNR(SRCNN, orijinal)
ssimCubic = ssim(bicubic, orijinal, multichannel = True)
ssimSR = ssim(SRCNN, orijinal, multichannel = True)
mseCubic = utilityFunc.MSE(bicubic, orijinal)
mseSR = utilityFunc.MSE(SRCNN, orijinal)

print("Bicubic_PSNR: {} \tSRCNN_PSNR: {}".format(psnrCubic, psnrSR))
print("Bicubic_SSIM: {} \tSRCNN_SSIM: {}".format(ssimCubic, ssimSR))
print("Bicubic_MSE: {} \tSRCNN_MSE: {}".format(mseCubic, mseSR))
