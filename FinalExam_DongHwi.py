#12214057 조동휘
import cv2
import numpy as np

#Load Video
cap = cv2.VideoCapture('./FinalExam_Clips.mp4')
if not cap.isOpened():
    print('Error Opening Video')

#Check Video Resolution
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

#initialize
current_frame = 0
show_frame = False
show_t = 0
bg_frame = 0
bg_count = 0
acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
bkg_sub = cv2.createBackgroundSubtractorMOG2()

#Run Video
while True:
    ret, frame = cap.read()
    if not ret:
        #Last Background Export
        cv2.imwrite('./Background{}.png'.format(bg_count), cv2.convertScaleAbs(acc_bgr/bg_frame))
        break

    #Show "Shot Changed" on Frame
    frame_c = frame.copy()
    if show_frame:
        show_t += 1
        cv2.putText(frame_c, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if show_t >= 60:
            show_t = 0
            show_frame = False

    #Check Histogram
    imgC = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Current Frame
    C_Hist = cv2.calcHist([imgC], [0], None, [256], [0, 256]) #Current Frame Histogram
    if not current_frame == 0:
        diff_Hist = C_Hist - P_Hist #Histogram substract
        # print('current_frame = {}, diff = {}'.format(current_frame, np.abs(diff_Hist).sum()))

        #diff_Hist over 100000 is shot changed
        if np.abs(diff_Hist).sum() > 100000:
            show_frame = True
            cv2.putText(frame_c, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            #if shot changed, Save Background Image
            cv2.imwrite('./Background{}.png'.format(bg_count), cv2.convertScaleAbs(acc_bgr/bg_frame))
            bkg_sub = cv2.createBackgroundSubtractorMOG2()
            acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
            bg_frame = 0
            bg_count += 1
        else:
            #background export
            bg_frame += 1
            blur = cv2.GaussianBlur(frame, (5, 5), 0.0)
            bImage = bkg_sub.apply(blur)
            ret, bImage = cv2.threshold(bImage, 20, 255, cv2.THRESH_BINARY_INV)
            cv2.accumulate(cv2.bitwise_and(frame, frame, mask=bImage), acc_bgr)

            #Optical Flow
            if current_frame != 0:
                contours, hierachy = cv2.findContours(cv2.bitwise_not(bImage), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                flow = cv2.calcOpticalFlowFarneback(imgP, imgC, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        x, y, width, height = cv2.boundingRect(cnt)
                        cv2.rectangle(frame_c, (x, y), (x+width, y+height), (0, 0, 255), 2)
                
    cv2.imshow('frame', frame_c)
    current_frame += 1
    imgP = imgC.copy() #Previous Frame
    P_Hist = C_Hist #Previous Frame Histogram
    key = cv2.waitKey(1)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()