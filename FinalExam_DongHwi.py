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
mask_f = np.zeros(shape=(height, width, 3), dtype=np.uint8)

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
        print('current_frame = {}, diff = {}'.format(current_frame, np.abs(diff_Hist).sum()))

        #diff_Hist over 100000 is shot changed
        if np.abs(diff_Hist).sum() > 100000:
            show_frame = True
            cv2.putText(frame_c, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            #if shot changed, Save Background Image
            cv2.imwrite('./Background{}.png'.format(bg_count), cv2.convertScaleAbs(acc_bgr/bg_frame))
            bkg_sub = cv2.createBackgroundSubtractorMOG2()
            acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
            mask_f = np.zeros(shape=(height, width, 3), dtype=np.uint8)
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
            if bg_frame != 0:
                corners_current = cv2.goodFeaturesToTrack(imgC, maxCorners=100, qualityLevel=0.05, minDistance=10)
                p1, st, err = cv2.calcOpticalFlowPyrLK(imgP, imgC, corners_current, None, winSize=(5, 5), maxLevel=3, criteria=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 10, 0.01))
                good_new = p1[st==1]
                good_old = corners_current[st==1]
                for i,(new,old) in enumerate(zip(good_new, good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask_f = cv2.line(mask_f, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2)
                    frame_c = cv2.circle(frame_c, (int(a), int(b)), 5, (255, 0, 0),-1)
                    frame_c = cv2.add(frame_c, mask_f)

    cv2.imshow('frame', frame_c)
    current_frame += 1
    imgP = imgC.copy() #Previous Frame
    P_Hist = C_Hist #Previous Frame Histogram
    key = cv2.waitKey(10)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()