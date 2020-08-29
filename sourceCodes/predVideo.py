import torch
import numpy as np
import model
import cv2
from timeit import default_timer as timer
import utilityFunc
from skimage.measure import compare_ssim as ssim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
net = model.Net()
net = net.to(device)

# use this for CPU usage
#net.load_state_dict(torch.load("net_epoch_120_CROP64_distFact_2.pth",map_location=lambda storage,loc:storage))
net.load_state_dict(torch.load("net_epoch_120_CROP64_distFact_2.pth"))

cap1 = cv2.VideoCapture(r'C:\Users\Casper\Desktop\n_orj.mp4')
cap2 = cv2.VideoCapture(r'C:\Users\Casper\Desktop\n_bulanik.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\Casper\Desktop\n_tahmin.mp4', fourcc, 23, (512, 512))

psnrCubicTotal, ssimCubicTotal, mseCubicTotal = 0, 0, 0
psnrSRCNNTotal, ssimSRCNNTotal, mseSRCNNTotal = 0, 0, 0 
font=cv2.FONT_HERSHEY_SIMPLEX
count = 0
it = 0
texts = "PSNR:34.23"
texts1 = "SSIM:0.87"
texts2 = "MSE:145.56"
psnr_list = []
ssim_list = []
mse_list = []
psnr_bicubic_list = []
#bottomLeftCornerOfText = (10,50)
#bottomLeftCornerOfText1=(10,70)
#bottomLeftCornerOfText2=(10,90)
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 0.5
#fontColor = (255,255,255)
lineType = 2

while True:
    ret1, frameOrg = cap1.read()
    ret2, frameDist = cap2.read()
    count += 1
    it += 1
    if ret1 == True:    
        #frameOrg = cv2.resize(frameOrg,(512,512),fx=0,fy=0, interpolation = cv2.INTER_AREA)

        LRframe_YCrCb = cv2.cvtColor(frameDist, cv2.COLOR_RGB2YCrCb)
        y = LRframe_YCrCb[:, :, 0]
        cb = LRframe_YCrCb[:, :, 1]
        cr = LRframe_YCrCb[:, :, 2]
        y = y.astype(float) / 255.
        print(y.shape)
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

        framePred = cv2.merge([pre, cb, cr])
        framePred = cv2.cvtColor(framePred, cv2.COLOR_YCrCb2RGB)
        
        #psnrCubic = utilityFunc.PSNR(frameDist, frameOrg)
        #psnrCubicTotal += psnrCubic
        #psnr_bicubic_list.append(psnrCubic)
        psnrSR = utilityFunc.PSNR(framePred, frameOrg)
        psnrSRCNNTotal += psnrSR
        psnr_list.append(psnrSR)
        #ssimCubic = ssim(frameDist, frameOrg, multichannel = True)
        #ssimCubicTotal += ssimCubic
        ssimSR = utilityFunc.ssim(framePred, frameOrg, multichannel = True)
        ssimSRCNNTotal += ssimSR
        ssim_list.append(ssimSR)
        #mseCubic = utilityFunc.MSE(frameDist, frameOrg)
        #mseCubicTotal += mseCubic
        mseSR = utilityFunc.MSE(framePred, frameOrg)
        mseSRCNNTotal += mseSRCNNTotal
        mse_list.append(mseSR)
        if count % 23 == 0:

            psnr=psnr_list[count-5]
            ssim=ssim_list[count-5]
            mse=mse_list[count-5]
            texts="PSNR:{:.2f}".format(psnr)
            texts1="SSIM:{:.2f}".format(ssim)
            texts2="MSE:{:.2f}".format(mse)

        cv2.putText(framePred,
                texts,
                (10, 50), 
                font, 
                fontScale,
                (0, 255, 0),
                lineType)

        cv2.putText(framePred,
                texts1,
                (10, 70), 
                font, 
                fontScale,
                (0, 255, 0),
                lineType)
        
        cv2.putText(framePred,
                texts2,
                (10, 90), 
                font, 
                fontScale,
                (0, 255, 0),
                lineType)
        out.write(framePred)
        
    else:
        break
  
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
