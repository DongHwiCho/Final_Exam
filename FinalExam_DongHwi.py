#12214057 조동휘
import cv2
import numpy as np

cap = cv2.VideoCapture('./FinalExam_Clips.mp4')
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
if not cap.isOpened():
    print('Error Opening Video')



current_frame = 0
show_frame = False
show_t = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if show_frame:
        show_t += 1
        cv2.putText(frame, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if show_t >= 60:
            show_t = 0
            show_frame = False

    imgC = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    C_Hist = cv2.calcHist([imgC], [0], None, [256], [0, 256])
    if not current_frame == 0:
        diff_Hist = C_Hist - P_Hist
        print('current_frame = {}, diff = {}'.format(current_frame, np.abs(diff_Hist).sum()))
        if np.abs(diff_Hist).sum() > 100000:
            show_frame = True
            cv2.putText(frame, 'Shot Changed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    current_frame += 1
    imgP = imgC.copy()
    P_Hist = cv2.calcHist([imgP], [0], None, [256], [0, 256])
    key = cv2.waitKey(10)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()