# Title: crop_bounds.py
# Purpose: To determine the correct cropping region for a print instance.
# Author: Cameron Palmer, campalme@clarkson.edu
# Last Modified: October 31st, 2022
#
# Code modified from https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/


import cv2
import json
import numpy as np
import os

FOLDER = 'E:/automation_dataset/images/ender_12'
IMG_EXT = '.jpg'
IMAGES_TO_CHECK = 5 # The number of images to take cropping data from. These are chosen randomly.
FINAL_CROP = 350    # Final size of square crop in pixels
OUTPUT_JSON = '_center.json'

SCALE = 4


refPt = []
cropped = False
ims = None
def click_and_crop(event, x, y, flags, param):
    global refPt, cropped
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    if not cropped:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropped = True
            # draw a rectangle around the region of interest
            cv2.rectangle(ims, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", ims)
            # construct the argument parser and parse the arguments

def crop(img_path): 
    global refPt, cropped
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(img_path)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    global ims
    ims = cv2.resize(image, (image.shape[1]//SCALE, image.shape[0]//SCALE))
    cv2.imshow("image", ims)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            ims = cv2.resize(image, (image.shape[1]//SCALE, image.shape[0]//SCALE))
            refPt = []
            cv2.imshow("image", ims)
            cropped = False

        # if 'c' key is pressed, confirm crop
        elif key == ord("c") and refPt != []:
            if len(refPt) == 2:
                roi = clone[refPt[0][1]*4:refPt[1][1]*SCALE, refPt[0][0]*4:refPt[1][0]*SCALE]
                cv2.imshow("ROI", roi)
            else:
                continue
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    cv2.destroyAllWindows()
                    corners = [(x*SCALE, y*SCALE) for (x,y) in refPt]
                    refPt = []
                    cropped = False
                    ims = None
                    return corners
                elif key == ord("r"):
                    cv2.destroyWindow("ROI")
                    image = clone.copy()
                    ims = cv2.resize(image, (image.shape[1]//SCALE, image.shape[0]//SCALE))
                    refPt = []
                    cv2.imshow("image", ims)
                    break
            


if __name__ == '__main__':
    

    files = os.listdir(FOLDER)

    images = [x for x in files if x.lower().endswith(IMG_EXT)]

    rand_images = np.random.choice(images, IMAGES_TO_CHECK)

    crops = []

    for img in rand_images:
        c = crop(os.path.join(FOLDER, img))
        print(c)
        crops.append(c)

    # Find all centers of the selected crops
    centers = []

    for crop in crops:
        center_x = (crop[0][0] + crop[1][0]) // 2
        center_y = (crop[0][1] + crop[1][1]) // 2
        centers.append((center_x, center_y))

    print(np.random.choice(images, 1))

    centers_x = [x for (x, y) in centers]
    centers_y = [y for (x, y) in centers]

    final_center = (float(np.average(centers_x)), float(np.average(centers_y)))

    print(final_center)

    image = cv2.imread(os.path.join(FOLDER, str(np.random.choice(images, 1)[0])))
    ims = cv2.resize(image, (image.shape[1]//SCALE, image.shape[0]//SCALE))

    # print(final_center[0])

    pt1 = [int(final_center[0] - FINAL_CROP), int(final_center[1] - FINAL_CROP)]
    pt2 = [int(final_center[0] + FINAL_CROP), int(final_center[1] + FINAL_CROP)]

    pt1_scaled = tuple(x//4 for x in pt1)
    pt2_scaled = tuple(x//4 for x in pt2)


    cv2.rectangle(ims, pt1_scaled, pt2_scaled, (0, 255, 0), 2)
    cv2.imshow('Example crop on random image', ims)
    cv2.waitKey(0)

    with open(os.path.join(FOLDER, OUTPUT_JSON), 'w') as f:
        json.dump(final_center, f)


    
