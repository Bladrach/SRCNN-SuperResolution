import cv2


# resize original video
cap = cv2.VideoCapture(r'C:\Users\Mehmet\Desktop\HighResPyTorch\video\niagara.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\Mehmet\Desktop\HighResPyTorch\video\n_orj.mp4', fourcc, 23, (512,512))
width = 512
height = 512
factor = 2
while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame, (512, 512), fx = 0, fy = 0, interpolation = cv2.INTER_AREA)
        out.write(b)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
