import torch
import numpy as np
import model
import cv2
import utilityFunc
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import glob


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
net = model.Net()
net = net.to(device)
net.load_state_dict(torch.load(".\\saved_models_CROP64_51200_2\\net_epoch_100.pth"))
size = 512
data_path = "C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\orijinalResimler\\label\\test_label\\*.png"
data_path = glob.glob(data_path)
for i in range(len(data_path)):
    y, cb, cr = utilityFunc.convertLR(data_path[i], '{}x{}OrijResim.png'.format(size, size), 'bulanikResim.png', factor = 2, size = size)
    y = torch.from_numpy(y).view(1, 1, y.shape[1], y.shape[0])
    print(y.shape)
    pre = net(y.to(device))
    print(pre.shape)
    pre = (pre.squeeze(0)).permute(1, 2, 0)
    pre = pre.cpu() 
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

    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(orijinal, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Orijinal')
    axs[1].imshow(cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Düşük Çözünürlük')
    axs[1].set_xlabel(xlabel = 'PSNR: {:.3f}\nMSE: {:.3f} \nSSIM: {:.3f}'.format(psnrCubic, mseCubic, ssimCubic))
    axs[2].imshow(cv2.cvtColor(SRCNN, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    axs[2].set_xlabel(xlabel = 'PSNR: {:.3f} \nMSE: {:.3f} \nSSIM: {:.3f}'.format(psnrSR, mseSR, ssimSR), color = "green")

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig('.\\demo\\demo{}.png'.format(i))
    plt.close()
    