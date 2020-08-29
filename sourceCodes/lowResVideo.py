import torch
import cv2
import utilityFunc
from skimage.measure import compare_ssim as ssim


cap = cv2.VideoCapture(r'C:\Users\Mehmet\Desktop\HighResPyTorch\video\n_orj.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\Mehmet\Desktop\HighResPyTorch\video\n_bulanik.mp4',fourcc, 23, (512,512))
width = 512
height = 512
factor = 2

font = cv2.FONT_HERSHEY_SIMPLEX

fontScale = 0.5
#fontColor = (255,255,255)
lineType = 2
psnrCubicTotal, ssimCubicTotal, mseCubicTotal = 0, 0, 0
psnrSRCNNTotal, ssimSRCNNTotal, mseSRCNNTotal = 0, 0, 0 
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
it = 0
texts = "PSNR:24.23"
texts1 = "SSIM:0.77"
texts2 = "MSE:241.27"

while True:
    ret1, frameOrg = cap.read()
    newWidth = int(width / factor)
    newHeight = int(height / factor)
    count+=1
    if ret1 == True:
        frame = cv2.resize(frameOrg, (newWidth, newHeight), cv2.INTER_CUBIC)
        frameDist = cv2.resize(frame, (512, 512), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
        
        if count % 23 == 0:
            psnr = psnrCubic
            #ssim=ssimCubic
            mse = mseCubic
            texts2 = "MSE:{:.2f}".format(mse)
            texts = "PSNR:{:.2f}".format(psnr)
            #texts1 = "SSIM:{:.2f}".format(ssim)
            
        cv2.putText(frameDist,
                texts,
                (10, 50), 
                font, 
                fontScale,
                (255, 255, 255),
                lineType)

        cv2.putText(frameDist,
                texts1,
                (10, 70), 
                font, 
                fontScale,
                (255, 255, 255),
                lineType)
        
        cv2.putText(frameDist,
                texts2,
                (10, 90), 
                font, 
                fontScale,
                (255, 255, 255),
                lineType)

        out.write(frameDist)
        psnrCubic = utilityFunc.PSNR(frameDist, frameOrg)
        psnrCubicTotal += psnrCubic
        #ssimCubic = ssim(frameDist, frameOrg, multichannel = True)
        #ssimCubicTotal += ssimCubic
        mseCubic = utilityFunc.MSE(frameDist, frameOrg)
        mseCubicTotal += mseCubic
      
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
