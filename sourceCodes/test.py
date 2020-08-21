import torch
import numpy as np
import model
import cv2
import utilityFunc
from skimage.measure import compare_ssim as ssim
import os
import glob
import matplotlib.pyplot as plt


data_path = glob.glob(r"C:\Users\Mehmet\Desktop\HighResPyTorch\orijinalResimler\label\test_label\\*.png")
dataNumber = len(data_path)
size = 512
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
net = model.Net()
net = net.to(device)
directory = "./saved_models"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.pth")))
files.sort(key=lambda x: os.path.getmtime(x))

psnrCubicTotal, ssimCubicTotal, mseCubicTotal = 0, 0, 0
psnrSRCNNTotal, ssimSRCNNTotal, mseSRCNNTotal = 0, 0, 0 
psnrCubic_list, ssimCubic_list, mseCubic_list = [], [], []
psnrSRCNN_list, ssimSRCNN_list, mseSRCNN_list = [], [], []
epoch_list = []
for filename in files:
    net.load_state_dict(torch.load(filename))
    print(filename)
    for i in range(dataNumber):
        print(i)
        y, cb, cr, orijinal, bicubic = utilityFunc.convertTestLR(data_path[i], factor = 2, size = size)
        y = torch.from_numpy(y).view(1, 1, y.shape[1], y.shape[0])
        pre = net(y.to(device))
        pre = (pre.squeeze(0)).permute(1, 2, 0)
        pre = pre.cpu()
        pre = pre.detach().numpy() * 255.
        #print(pre.shape)
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        pre = pre[:, :, 0]
        #print(pre.shape)
        #print(cb.shape)
        #print(cr.shape)
        img = cv2.merge([pre, cb, cr])
        SRCNN = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        psnrCubic = utilityFunc.PSNR(bicubic, orijinal)
        psnrSR = utilityFunc.PSNR(SRCNN, orijinal)
        ssimCubic = ssim(bicubic, orijinal, multichannel = True)
        ssimSR = ssim(SRCNN, orijinal, multichannel = True)
        mseCubic = utilityFunc.MSE(bicubic, orijinal)
        mseSR = utilityFunc.MSE(SRCNN, orijinal)
        psnrCubicTotal += psnrCubic
        ssimCubicTotal += ssimCubic
        mseCubicTotal += mseCubic
        psnrSRCNNTotal += psnrSR
        ssimSRCNNTotal += ssimSR
        mseSRCNNTotal += mseSR
    psnrCubic_list.append(psnrCubicTotal / dataNumber)
    ssimCubic_list.append(ssimCubicTotal / dataNumber)
    mseCubic_list.append(mseCubicTotal / dataNumber)
    psnrSRCNN_list.append(psnrSRCNNTotal / dataNumber)
    ssimSRCNN_list.append(ssimSRCNNTotal / dataNumber)
    mseSRCNN_list.append(mseSRCNNTotal / dataNumber)
    psnrCubicTotal, ssimCubicTotal, mseCubicTotal = 0, 0, 0
    psnrSRCNNTotal, ssimSRCNNTotal, mseSRCNNTotal = 0, 0, 0 
    epoch_list.append(int(filename.split("_")[-1].split(".")[0]))

fig = plt.figure()
plt.subplot(1,3,1)
plt.plot(epoch_list, psnrCubic_list, label = 'Bicubic')
plt.plot(epoch_list, psnrSRCNN_list, label = 'SRCNN')
plt.title("PSNR ({:.3f} dB)".format(float(max(psnrSRCNN_list))))
plt.legend()
plt.grid()
plt.xticks(range(0, 101, 10))
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")

plt.subplot(1,3,2)
plt.plot(epoch_list, mseCubic_list, label = 'Bicubic')
plt.plot(epoch_list, mseSRCNN_list, label = 'SRCNN')
plt.title("MSE ({:.3f})".format(float(min(mseSRCNN_list))))
plt.legend()
plt.grid()
plt.xticks(range(0, 101, 10))
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.subplot(1,3,3)
plt.plot(epoch_list, ssimCubic_list, label = 'Bicubic')
plt.plot(epoch_list, ssimSRCNN_list, label = 'SRCNN')
plt.title("SSIM ({:.3f})".format(float(max(ssimSRCNN_list))))
plt.legend()
plt.grid()
plt.xticks(range(0, 101, 10))
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.tight_layout(pad = 1.3)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
fig.savefig("Test.jpg")
plt.show()
