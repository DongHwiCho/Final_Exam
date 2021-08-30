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
bkg_sub = np.zeros(shape=(height, width, 3), dtype=np.uint8)

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
            acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
            bg_frame = 0
            bg_count += 1
        else:
            #background export
            bg_frame += 1
            cv2.accumulate(frame, acc_bgr)


    TH = 40
    bkg_bgr = acc_bgr/bg_frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0.0)
    diff_bgr = np.uint8(cv2.absdiff(np.float32(blur), bkg_bgr))
    db, dg, dr = cv2.split(diff_bgr)
    ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY)
    ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY)
    ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY)
    bImage = cv2.bitwise_or(bb, bg)
    bImage = cv2.bitwise_or(br, bImage)
    bImage = cv2.erode(bImage, None, 5)
    bImage = cv2.dilate(bImage, None, 5)
    bImage = cv2.erode(bImage, None, 7)
    cv2.imshow('bImage', bImage)
    msk = bImage.copy()

    msk = cv2.bitwise_not(msk)
    cv2.bitwise_or(bkg_sub, frame, mask=msk)

    cv2.imshow('bImage', bkg_sub)
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