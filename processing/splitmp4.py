import cv2
import os


INPUT_MP4 = '../_data\mp4s\Z1.8-Y--4.26_20221031190036.mp4'
OUTPUT_DIR = '../_data'
OUTPUT_FOLDER = 'ender_0'


if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

capture = cv2.VideoCapture(INPUT_MP4)
 
frameNr = 0

while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER, str(OUTPUT_FOLDER) + '_' + str(frameNr) + '.jpg'), frame)
 
    else:
        break
 
    frameNr += 1
 
capture.release()