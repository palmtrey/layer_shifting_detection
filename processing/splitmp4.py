import cv2
import os


INPUT_MP4 = 'E:/automation_dataset/mp4s/Z3.9-X--4.04_20221102144912.mp4'
OUTPUT_DIR = 'E:/automation_dataset/images'
OUTPUT_FOLDER = 'ender_3'


if not os.path.isdir(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER)):
    os.mkdir(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER))

capture = cv2.VideoCapture(INPUT_MP4)
 
frameNr = 0

while (True):
    print('frame')
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER, str(OUTPUT_FOLDER) + '_' + str(frameNr) + '.jpg'), frame)
 
    else:
        break
 
    frameNr += 1
 
capture.release()