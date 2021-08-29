#12214057 조동휘
import cv2
import numpy as np

#initialize
current_frame = 0
show_frame = False
show_t = 0

#Load Video
cap = cv2.VideoCapture('./FinalExam_Clips.mp4')
if not cap.isOpened():
    print('Error Opening Video')

#Check Video Resolution
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

#Run Video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Show "Shot Changed" on Frame
    if show_frame:
        show_t += 1
        cv2.putText(frame, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if show_t >= 60:
            show_t = 0
            show_frame = False

    #Check Histogram
    imgC = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Current Frame
    C_Hist = cv2.calcHist([imgC], [0], None, [256], [0, 256]) #Current Frame Histogram
    if not current_frame == 0:
        diff_Hist = C_Hist - P_Hist #Histogram substract
        print('current_frame = {}, diff = {}'.format(current_frame, np.abs(diff_Hist).sum()))
        
        #diff_Hist over 100000 is shot changed
        if np.abs(diff_Hist).sum() > 100000:
            show_frame = True
            cv2.putText(frame, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    current_frame += 1
    imgP = imgC.copy() #Previous Frame
    P_Hist = C_Hist #Previous Frame Histogram
    key = cv2.waitKey(10)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()