import cv2
import os


INPUT_MP4 = '../_data/mp4s/correct_phil_28m_20221031180938.mp4'
OUTPUT_DIR = '../_data/correct_phil_ECEender3v2_cam0_Oct31_0'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

capture = cv2.VideoCapture(INPUT_MP4)
 
frameNr = 0

while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(os.path.join(OUTPUT_DIR, str(frameNr) + '.jpg'), frame)
 
    else:
        break
 
    frameNr += 1
 
capture.release()